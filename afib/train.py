"""Reproducible outer-fold training with untouched test patients until selection."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import subprocess
import time

import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)
import torch
from torch.utils.data import DataLoader

from afib import HRV_ORDER
from afib.data import AFibDataset, HRVTransform, load_arrays, patient_splits
from afib.models import AFibModel
from afib.preprocess import external_root


def evaluate(model, loader, device):
    model.eval()
    labels, logits = [], []
    with torch.no_grad():
        for ecg, hrv, target in loader:
            prediction = model(ecg.to(device), hrv.to(device)).cpu()
            if not torch.isfinite(prediction).all():
                raise RuntimeError('Nonfinite evaluation logits')
            labels.extend(target.tolist())
            logits.extend(prediction.tolist())
    y, z = torch.tensor(labels), torch.tensor(logits)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y).item()
    return loss, np.asarray(labels, dtype=int), torch.sigmoid(z).numpy()


def metrics(labels, probabilities):
    predictions = probabilities >= .5
    both = len(set(labels)) == 2
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    return dict(n=len(labels), positives=int(sum(labels)), threshold=.5,
        accuracy=float(accuracy_score(labels, predictions)),
        precision=float(precision_score(labels, predictions, zero_division=0)),
        recall=float(recall_score(labels, predictions, zero_division=0)),
        f1=float(f1_score(labels, predictions, zero_division=0)),
        specificity=float(tn/(tn+fp)) if tn+fp else None,
        auroc=float(roc_auc_score(labels, probabilities)) if both else None,
        average_precision=float(average_precision_score(labels, probabilities)) if both else None,
        confusion_matrix=[[int(tn), int(fp)], [int(fn), int(tp)]])


def train_fold(rows, fold, indices, out, args, manifest_hash):
    seed = args.seed + fold
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_idx, val_idx, test_idx = indices
    partitions = [[rows[int(i)] for i in ids] for ids in indices]
    train_rows, val_rows, test_rows = partitions
    directory = out / f'{args.model}-fold{fold}'
    directory.mkdir(parents=True, exist_ok=False)
    split = {name: dict(patients=sorted({r['patient'] for r in part}),
                       labels=dict(Counter(r['label'] for r in part)), indices=ids.tolist())
             for name, part, ids in zip(('train', 'validation', 'test'), partitions, indices)}
    (directory / 'split.json').write_text(json.dumps(split, indent=2))
    transform = HRVTransform().fit(np.stack([load_arrays(r)[1] for r in train_rows]))
    train_loader = DataLoader(AFibDataset(train_rows, transform), batch_size=args.batch_size,
                              shuffle=True, generator=torch.Generator().manual_seed(seed), num_workers=0)
    val_loader = DataLoader(AFibDataset(val_rows, transform), batch_size=args.batch_size, num_workers=0)
    model = AFibModel(se=args.model == 'se').to(device)
    parameter_count = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    positives = sum(r['label'] for r in train_rows)
    pos_weight = (len(train_rows)-positives) / positives
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))
    config = dict(vars(args), device=str(device), parameter_count=parameter_count,
        manifest_sha256=manifest_hash, torch_version=str(torch.__version__),
        git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        fold_seed=seed, hrv_order=HRV_ORDER, pos_weight=pos_weight)
    (directory / 'config.json').write_text(json.dumps(config, indent=2))
    print(json.dumps(dict(fold=fold, device=str(device), parameters=parameter_count,
                          split={k: {n:v for n,v in s.items() if n != 'indices'} for k,s in split.items()})), flush=True)
    best, stale, history = float('inf'), 0, []
    initial_state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    for epoch in range(args.epochs):
        started = time.time()
        model.train()
        total, count = 0., 0
        for step, (ecg, hrv, target) in enumerate(train_loader):
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(ecg.to(device), hrv.to(device)), target.to(device))
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite training loss')
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            total += loss.item() * len(target)
            count += len(target)
            if args.max_steps and step + 1 >= args.max_steps:
                break
        val_loss, y, probabilities = evaluate(model, val_loader, device)
        entry = dict(epoch=epoch+1, train_loss=total/count, validation_loss=val_loss,
                     validation=metrics(y, probabilities), seconds=time.time()-started,
                     last_gradient_norm=float(norm))
        history.append(entry)
        (directory / 'history.json').write_text(json.dumps(history, indent=2))
        print(json.dumps(dict(fold=fold, **entry)), flush=True)
        if val_loss < best:
            best, stale = val_loss, 0
            torch.save(dict(model=model.state_dict(), transform=transform.state(),
                            config=config, epoch=epoch+1, validation_loss=best,
                            selection='minimum inner-validation unweighted BCE',
                            optimizer=optimizer.state_dict()), directory / 'best.pt')
        else:
            stale += 1
        if stale >= args.patience:
            break
    checkpoint = torch.load(directory / 'best.pt', map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model'])
    changed = any(not torch.equal(value.cpu(), initial_state[key]) for key,value in model.state_dict().items())
    if not changed:
        raise RuntimeError('Checkpoint identical to initialization')
    # Smoke runs deliberately never inspect outer-test outcomes.
    if args.max_steps:
        print(f'Smoke checkpoint verified: {directory / "best.pt"}; outer test untouched', flush=True)
        return
    test_loader = DataLoader(AFibDataset(test_rows, transform), batch_size=args.batch_size, num_workers=0)
    test_loss, labels, probabilities = evaluate(model, test_loader, device)
    result = dict(fold=fold, selected_epoch=checkpoint['epoch'], test_loss=test_loss,
                  test=metrics(labels, probabilities))
    (directory / 'test_metrics.json').write_text(json.dumps(result, indent=2))
    predictions = [dict(patient=row['patient'], start_sample=row['start_sample'], label=int(label),
                        probability=float(prob)) for row,label,prob in zip(test_rows, labels, probabilities)]
    (directory / 'test_predictions.json').write_text(json.dumps(predictions, indent=2))
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--run', required=True)
    parser.add_argument('--model', choices=['reference', 'se'], default='se')
    parser.add_argument('--folds', nargs='+', type=int, default=[0])
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--max-steps', type=int, default=0)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or any(f not in range(5) for f in args.folds):
        raise ValueError('Invalid training arguments')
    if Path(args.run).name != args.run:
        raise ValueError('Run must be a directory name, not a path')
    root = external_root(args.root)
    manifest = root / 'manifest.json'
    rows = json.loads(manifest.read_text())
    manifest_hash = hashlib.sha256(manifest.read_bytes()).hexdigest()
    out = root / 'runs' / args.run
    out.mkdir(parents=True, exist_ok=True)
    for fold, train, val, test in patient_splits(rows, args.seed):
        if fold in args.folds:
            train_fold(rows, fold, (train, val, test), out, args, manifest_hash)


if __name__ == '__main__':
    main()

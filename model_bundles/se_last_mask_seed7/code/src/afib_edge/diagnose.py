"""Training-only cache audit and fixed-subset memorization diagnostic."""
from __future__ import annotations
import argparse
import csv
import json
import time
from pathlib import Path
import numpy as np
import torch
from .dataset import CacheWindowDataset
from .hrv_preprocess import HRVFoldPreprocessor
from .models import ModelConfig, build_model
from .training import make_outer_fold_indices, make_validation_split, resolve_device


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache-root', type=Path, required=True)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--device', default='cuda')
    p.add_argument('--steps', type=int, default=200)
    p.add_argument('--fold', type=int, default=0)
    a = p.parse_args()
    if a.steps < 1:
        p.error('--steps must be positive')
    a.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    def log(message):
        message = f'[{time.monotonic()-started:.1f}s] {message}'
        print(message, flush=True)
        with (a.output_dir / 'progress.log').open('a') as f:
            f.write(message+'\n')
    torch.manual_seed(7)
    np.random.seed(7)
    device = resolve_device(a.device)
    d = CacheWindowDataset(a.cache_root, a.manifest)
    outer, test = make_outer_fold_indices(d.labels, d.subject_ids, fold=a.fold)
    train, val = make_validation_split(outer, d.subject_ids)
    groups = [set(d.subject_ids[x]) for x in (train, val, test)]
    assert not (groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
    report = {'fold': a.fold, 'scope': 'inner training only; no validation/test inference',
              'group_identity_caveat': 'subject_id is database:record; cross-record patient identity requires source verification',
              'inner_train_windows': len(train), 'inner_train_records': len(groups[0]),
              'outer_test_accessed': False, 'records': []}
    log('Checking every inner-training ECG window and cache/manifest alignment')
    for subject in sorted(groups[0]):
        indices = train[d.subject_ids[train] == subject]
        rms, flat = [], 0
        for i in indices:
            x = d[int(i)]['ecg'].numpy()
            std = x.std(axis=-1)
            flat += int((std < 1e-6).sum())
            rms.append(float(np.sqrt(np.mean(x.astype(np.float64)**2))))
        h = d.raw_hrv_for_indices(indices)
        entry = {'subject_id': subject, 'windows': len(indices),
                 'positives': int(d.labels[indices].sum()), 'flat_segments': flat,
                 'ecg_rms_min': min(rms), 'ecg_rms_median': float(np.median(rms)),
                 'ecg_rms_max': max(rms),
                 'hrv_missing_fraction': np.mean(~np.isfinite(h), axis=(0,1)).tolist()}
        report['records'].append(entry)
        log(f"Checked {subject}: {len(indices)} windows, {entry['positives']} positives")
    d.hrv_preprocessor = HRVFoldPreprocessor().fit(d.raw_hrv_for_indices(train))
    report['hrv_preprocessor'] = d.hrv_preprocessor.state_dict()
    (a.output_dir/'audit.json').write_text(json.dumps(report, indent=2))
    # Deterministically pick four windows per class, spreading across records.
    rng = np.random.default_rng(7)
    chosen = []
    for label in (0, 1):
        candidates = rng.permutation(train[d.labels[train] == label])
        selected, seen = [], set()
        for i in candidates:
            subject = d.subject_ids[i]
            if subject not in seen:
                selected.append(int(i)); seen.add(subject)
            if len(selected) == 4:
                break
        if len(selected) < 4:
            raise ValueError('Need four distinct inner-training records per class for this diagnostic')
        chosen.extend(selected)
    batches = [d[i] for i in chosen]
    ecg = torch.stack([b['ecg'] for b in batches]).to(device)
    hrv = torch.stack([b['hrv'] for b in batches]).to(device)
    y = torch.stack([b['label'] for b in batches]).to(device)
    report['selected_windows'] = [vars(d.window_refs[i]) for i in chosen]
    report['memorization'] = []
    # Both arms are diagnostics, not a validation-selected production change.
    for normalize in (False, True):
        torch.manual_seed(7)
        model = build_model('proposed', ModelConfig(dropout=0.0)).to(device)
        x = ecg.clone()
        if normalize:
            x = (x-x.mean(dim=-1, keepdim=True))/x.std(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-6)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0)
        history = []
        log(f'Start fixed 8-window diagnostic: segment_normalization={normalize}; BCE; dropout=0')
        for step in range(1, a.steps+1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(model(x, hrv), y)
            if not torch.isfinite(loss):
                raise ValueError('Non-finite diagnostic loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if step == 1 or step % 10 == 0 or step == a.steps:
                model.eval()
                with torch.no_grad():
                    logits = model(x, hrv)
                    eval_loss = float(torch.nn.functional.binary_cross_entropy_with_logits(logits, y))
                    accuracy = float(((logits >= 0) == y.bool()).float().mean())
                history.append({'step': step, 'train_loss': float(loss.detach()), 'eval_loss': eval_loss, 'accuracy': accuracy})
                log(f'normalized={normalize} step {step}/{a.steps}: train_BCE={float(loss.detach()):.4f} eval_BCE={eval_loss:.4f} accuracy={accuracy:.3f}')
                if accuracy == 1.0 and eval_loss < 0.05:
                    break
        report['memorization'].append({'segment_normalization': normalize, 'history': history,
                                      'passed': accuracy == 1.0 and eval_loss < 0.05})
        (a.output_dir/'diagnostic_summary.json').write_text(json.dumps(report, indent=2))
        del model, optimizer
    log('FINISHED. Send diagnostic_summary.json; these are training diagnostics, not generalization scores.')

if __name__ == '__main__':
    main()

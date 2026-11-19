from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .model import CurrentAFDetector


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


class Segments(Dataset):
    def __init__(self, root: Path, indices: np.ndarray) -> None:
        self.ecg = np.load(root / "ecg.npy", mmap_mode="r")
        self.labels = np.load(root / "labels.npy", mmap_mode="r")
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int):
        index = int(self.indices[item])
        signal = np.asarray(self.ecg[index], dtype=np.float32).copy()
        signal = (signal - signal.mean()) / max(float(signal.std()), 1e-6)
        return torch.from_numpy(signal[None, :]), torch.tensor(float(self.labels[index])), index


def predict(model, loader, device):
    model.eval()
    ys, ps, ids = [], [], []
    with torch.no_grad():
        for x, y, index in loader:
            logits = model(x.to(device, non_blocking=device.type == "cuda"))
            ys.append(y.numpy())
            ps.append(torch.sigmoid(logits).cpu().numpy())
            ids.append(index.numpy())
    return np.concatenate(ys), np.concatenate(ps), np.concatenate(ids)


def choose_threshold(y: np.ndarray, p: np.ndarray, specificity_floor: float) -> float:
    negatives = p[y == 0]
    if len(negatives) == 0:
        raise ValueError("validation split has no negative samples")
    candidates = np.unique(p)
    feasible = []
    for threshold in candidates:
        prediction = p >= threshold
        tn = int(((prediction == 0) & (y == 0)).sum())
        fp = int(((prediction == 1) & (y == 0)).sum())
        tp = int(((prediction == 1) & (y == 1)).sum())
        fn = int(((prediction == 0) & (y == 1)).sum())
        specificity = tn / max(tn + fp, 1)
        recall = tp / max(tp + fn, 1)
        if specificity >= specificity_floor:
            feasible.append((recall, specificity, -float(threshold), float(threshold)))
    return max(feasible)[-1] if feasible else 1.0


def confusion(y, p, threshold):
    pred = p >= threshold
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    return {
        "true_positives": tp, "true_negatives": tn,
        "false_positives": fp, "false_negatives": fn,
        "recall": tp / max(tp + fn, 1),
        "specificity": tn / max(tn + fp, 1),
        "precision": tp / max(tp + fp, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--specificity-floor", type=float, default=0.95)
    parser.add_argument("--device", choices=("cuda", "cpu", "auto"), default="cuda")
    args = parser.parse_args()

    required = [args.cache_root / name for name in ("ecg.npy", "labels.npy", "patient_ids.npy", "metadata.json", "COMPLETE")]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing Icentia cache inputs:\n" + "\n".join(missing))
    ecg = np.load(args.cache_root / "ecg.npy", mmap_mode="r")
    labels = np.load(args.cache_root / "labels.npy")
    groups = np.load(args.cache_root / "patient_ids.npy").astype(str)
    if ecg.shape != (len(labels), 7500) or len(labels) != len(groups):
        raise ValueError(f"unexpected cache arrays: ECG={ecg.shape}, labels={labels.shape}, patients={groups.shape}")
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("labels must be binary 0/1")

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    splitter = GroupKFold(n_splits=5)
    predictions = []
    fold_rows = []
    seen_test: list[int] = []
    started = time.monotonic()

    for fold, (development, test) in enumerate(splitter.split(np.arange(len(labels)), labels, groups)):
        checkpoint_path = args.output_dir / f"fold{fold}_current_af.pt"
        prediction_path = args.output_dir / f"fold{fold}_test_predictions.csv"
        if checkpoint_path.is_file() and prediction_path.is_file():
            saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            if int(saved.get("seed", -1)) != args.seed or float(saved.get("specificity_floor", -1)) != args.specificity_floor:
                raise ValueError(f"fold {fold} resume settings changed; use a new output directory")
            fold_prediction = pd.read_csv(prediction_path)
            if set(map(int, fold_prediction.sample_index)) != set(map(int, test)):
                raise ValueError(f"fold {fold} resume predictions do not match the locked test indices")
            metrics = saved["metrics"]
            fold_rows.append({
                "fold": fold, "best_epoch": int(saved["best_epoch"]),
                "threshold": float(saved["threshold"]), **metrics,
                "checkpoint_sha256": sha256(checkpoint_path),
            })
            predictions.append(fold_prediction)
            seen_test.extend(map(int, fold_prediction.sample_index))
            print(f"RESUME: fold {fold} already complete", flush=True)
            continue
        inner = GroupShuffleSplit(n_splits=100, test_size=0.2, random_state=args.seed + fold)
        train = validation = None
        for candidate_train, candidate_validation in inner.split(development, labels[development], groups[development]):
            a, b = development[candidate_train], development[candidate_validation]
            if len(np.unique(labels[a])) == 2 and len(np.unique(labels[b])) == 2:
                train, validation = a, b
                break
        if train is None:
            raise RuntimeError(f"could not form a two-class inner split for fold {fold}")
        if set(groups[train]) & set(groups[validation]) or set(groups[development]) & set(groups[test]):
            raise AssertionError("patient leakage")

        torch.manual_seed(args.seed + fold)
        np.random.seed(args.seed + fold)
        random.seed(args.seed + fold)
        model = CurrentAFDetector().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
        counts = np.bincount(labels[train].astype(int), minlength=2)
        positive_weight = torch.tensor(float(counts[0] / max(counts[1], 1)), device=device)
        generator = torch.Generator().manual_seed(args.seed + fold)
        train_loader = DataLoader(Segments(args.cache_root, train), batch_size=args.batch_size, shuffle=True, generator=generator, pin_memory=device.type == "cuda")
        validation_loader = DataLoader(Segments(args.cache_root, validation), batch_size=args.batch_size, shuffle=False, pin_memory=device.type == "cuda")
        test_loader = DataLoader(Segments(args.cache_root, test), batch_size=args.batch_size, shuffle=False, pin_memory=device.type == "cuda")
        best_auc, best_epoch, stale, best_state = -1.0, 0, 0, None
        history = []
        for epoch in range(1, args.epochs + 1):
            model.train()
            total, count = 0.0, 0
            for x, y, _ in train_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = nn.functional.binary_cross_entropy_with_logits(model(x), y, pos_weight=positive_weight)
                if not torch.isfinite(loss):
                    raise FloatingPointError("non-finite detector loss")
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total += float(loss.detach()) * len(y)
                count += len(y)
            yv, pv, _ = predict(model, validation_loader, device)
            auc = float(roc_auc_score(yv, pv))
            ap = float(average_precision_score(yv, pv))
            history.append({"epoch": epoch, "train_loss": total / count, "validation_auroc": auc, "validation_ap": ap})
            if auc > best_auc + 1e-6:
                best_auc, best_epoch, stale = auc, epoch, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                stale += 1
            print(f"fold={fold} epoch={epoch:02d} val_AUROC={auc:.4f} val_AP={ap:.4f} stale={stale}/{args.patience}", flush=True)
            if stale >= args.patience:
                break
        if best_state is None:
            raise RuntimeError("no detector checkpoint")
        model.load_state_dict(best_state)
        yv, pv, _ = predict(model, validation_loader, device)
        threshold = choose_threshold(yv, pv, args.specificity_floor)
        yt, pt, it = predict(model, test_loader, device)
        metrics = confusion(yt, pt, threshold)
        metrics.update({"auroc": float(roc_auc_score(yt, pt)), "average_precision": float(average_precision_score(yt, pt))})
        checkpoint = {
            "format": "current_af_detector_fold_v1", "fold": fold,
            "seed": args.seed,
            "model_state_dict": best_state, "model_config": model.checkpoint_config(),
            "parameter_count": model.parameter_count, "threshold": threshold,
            "specificity_floor": args.specificity_floor, "best_epoch": best_epoch,
            "train_patients": sorted(set(groups[train])),
            "validation_patients": sorted(set(groups[validation])),
            "test_patients": sorted(set(groups[test])), "metrics": metrics,
        }
        torch.save(checkpoint, checkpoint_path)
        fold_rows.append({"fold": fold, "best_epoch": best_epoch, "threshold": threshold, **metrics, "checkpoint_sha256": sha256(checkpoint_path)})
        fold_prediction = pd.DataFrame({"sample_index": it, "patient_id": groups[it], "label": yt.astype(int), "probability": pt, "threshold": threshold, "fold": fold})
        fold_prediction.to_csv(prediction_path, index=False)
        predictions.append(fold_prediction)
        seen_test.extend(map(int, it))
        print(f"FOLD {fold} LOCKED TEST: AUROC={metrics['auroc']:.4f} AP={metrics['average_precision']:.4f} recall={metrics['recall']:.4f} specificity={metrics['specificity']:.4f}", flush=True)

    oof = pd.concat(predictions, ignore_index=True).sort_values("sample_index")
    oof.to_csv(args.output_dir / "oof_predictions.csv", index=False)
    y = oof.label.to_numpy()
    p = oof.probability.to_numpy()
    pred = p >= oof.threshold.to_numpy()
    aggregate = confusion(y, pred.astype(float), 0.5)
    aggregate.update({"auroc": float(roc_auc_score(y, p)), "average_precision": float(average_precision_score(y, p))})
    unique, counts = np.unique(np.asarray(seen_test), return_counts=True)
    all_once = len(unique) == len(labels) and bool(np.all(counts == 1))
    parameter_count = CurrentAFDetector().parameter_count
    summary = {
        "format": "current_af_detector_nested_group_cv_v1",
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "samples": int(len(labels)), "patients": int(len(np.unique(groups))),
        "positive_samples": int(labels.sum()), "negative_samples": int((labels == 0).sum()),
        "parameter_count": parameter_count, "forecast_plus_detector_parameters": 159503 + parameter_count,
        "specificity_floor": args.specificity_floor, "folds": fold_rows,
        "oof": aggregate, "elapsed_seconds": time.monotonic() - started,
        "gates": {
            "all_samples_tested_once": all_once,
            "five_folds_complete": len(fold_rows) == 5,
            "patient_disjoint": True,
            "parameter_budget_combined_under_250k": 159503 + parameter_count < 250000,
            "finite_oof_metrics": bool(np.isfinite([aggregate["auroc"], aggregate["average_precision"]]).all()),
        },
    }
    summary["all_gates_pass"] = all(summary["gates"].values())
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print("\nCURRENT-AF DETECTOR FIVE-FOLD EVALUATION COMPLETE")


if __name__ == "__main__":
    main()

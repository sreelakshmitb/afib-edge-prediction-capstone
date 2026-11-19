"""Resumable fold-0 validation-only pilot; outer-test patients are never loaded."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset

from afib_unified.dataset import UnifiedCacheDataset
from afib_unified.hrv_preprocess import HRVFoldPreprocessor
from afib_unified.manifest import sha256_file
from afib_unified.models import build_model
from afib_unified.sampling import HierarchicalSamplerConfig, build_epoch_schedule


def classification(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict:
    predicted = probabilities >= threshold
    positive = labels == 1
    tp = int(np.sum(predicted & positive))
    tn = int(np.sum(~predicted & ~positive))
    fp = int(np.sum(predicted & ~positive))
    fn = int(np.sum(~predicted & positive))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "threshold": float(threshold), "true_positives": tp, "true_negatives": tn,
        "false_positives": fp, "false_negatives": fn, "precision": precision,
        "recall": recall, "specificity": specificity, "f1": f1,
        "accuracy": (tp + tn) / len(labels),
    }


def select_threshold(labels: np.ndarray, probabilities: np.ndarray) -> dict:
    """Predeclared rule: maximize recall subject to at least 90% specificity."""
    candidates = np.unique(np.concatenate(([0.0], probabilities, [1.0])))
    eligible = []
    for threshold in candidates:
        metrics = classification(labels, probabilities, float(threshold))
        if metrics["specificity"] >= 0.90:
            eligible.append(metrics)
    if not eligible:
        raise RuntimeError("no validation threshold reaches 90% specificity")
    eligible.sort(
        key=lambda item: (item["recall"], item["f1"], item["specificity"], -item["threshold"]),
        reverse=True,
    )
    return eligible[0]


def evaluate(model, data, indices, device, batch_size=32):
    loader = DataLoader(
        Subset(data, [int(i) for i in indices]), batch_size=batch_size,
        shuffle=False, num_workers=0, pin_memory=True,
    )
    model.eval()
    labels, probabilities, manifest_indices = [], [], []
    total_loss = 0.0
    with torch.no_grad():
        for batch in loader:
            y = batch["label"].to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(
                    batch["ecg"].to(device, non_blocking=True),
                    batch["hrv"].to(device, non_blocking=True),
                )
                loss = F.binary_cross_entropy_with_logits(logits, y)
            total_loss += float(loss) * len(y)
            labels.append(y.cpu().numpy())
            probabilities.append(torch.sigmoid(logits).float().cpu().numpy())
            manifest_indices.extend(int(value) for value in batch["manifest_index"])
    y = np.concatenate(labels).astype(np.uint8)
    p = np.concatenate(probabilities).astype(np.float64)
    ranking = {
        "loss": total_loss / len(y),
        "pr_auc": float(average_precision_score(y, p)),
        "auroc": float(roc_auc_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "positive_count": int(y.sum()),
        "window_count": int(len(y)),
    }
    return ranking, y, p, np.asarray(manifest_indices, dtype=np.int64)


def save_checkpoint(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy-cache", type=Path, required=True)
    parser.add_argument("--iridia-cache", type=Path, required=True)
    parser.add_argument("--protocol-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batches-per-epoch", type=int, default=50)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    seed = 7
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    manifest_path = args.protocol_dir / "unified_cached_manifest.csv"
    splits_path = args.protocol_dir / "locked_patient_splits.json"
    protocol_summary = json.loads((args.protocol_dir / "protocol_summary.json").read_text())
    if sha256_file(manifest_path) != protocol_summary["manifest_sha256"]:
        raise ValueError("unified manifest hash mismatch")
    if sha256_file(splits_path) != protocol_summary["splits_sha256"]:
        raise ValueError("frozen split hash mismatch")
    manifest = pd.read_csv(manifest_path).sort_values("manifest_index").reset_index(drop=True)
    split_data = json.loads(splits_path.read_text())
    fold = next(row for row in split_data["folds"] if int(row["fold"]) == args.fold)
    train_subjects = set(fold["train_subjects"])
    validation_subjects = set(fold["validation_subjects"])
    test_subjects = set(fold["test_subjects"])
    if train_subjects & validation_subjects or (train_subjects | validation_subjects) & test_subjects:
        raise AssertionError("patient leakage in frozen split")
    train_indices = manifest.index[manifest.subject_id.astype(str).isin(train_subjects)].to_numpy()
    validation_indices = manifest.index[manifest.subject_id.astype(str).isin(validation_subjects)].to_numpy()
    # The test indices are deliberately never constructed or passed to the dataset sampler.
    args.output_dir.mkdir(parents=True, exist_ok=True)
    last_path = args.output_dir / "last_checkpoint.pt"
    best_path = args.output_dir / "best_checkpoint.pt"
    history_path = args.output_dir / "history.json"
    device = torch.device("cuda")
    raw = UnifiedCacheDataset(
        args.legacy_cache, args.iridia_cache, manifest_path, None, "none"
    )
    checkpoint = None
    if args.resume and last_path.is_file():
        checkpoint = torch.load(last_path, map_location="cpu", weights_only=False)
        if checkpoint["manifest_sha256"] != protocol_summary["manifest_sha256"]:
            raise ValueError("resume checkpoint belongs to a different manifest")
        preprocessor = HRVFoldPreprocessor.from_state_dict(checkpoint["hrv_preprocessor"])
        print(f"RESUME from epoch {checkpoint['epoch'] + 1}", flush=True)
    else:
        print(f"Fit HRV preprocessing on {len(train_indices)} training windows only", flush=True)
        preprocessor = HRVFoldPreprocessor().fit(raw.raw_hrv_for_indices(train_indices))
    data = UnifiedCacheDataset(
        args.legacy_cache, args.iridia_cache, manifest_path,
        preprocessor, "segment_zscore",
    )
    model = build_model("proposed").to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda")
    start_epoch, best_pr_auc, stale_epochs, history = 0, -math.inf, 0, []
    if checkpoint is not None:
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_pr_auc = float(checkpoint["best_pr_auc"])
        stale_epochs = int(checkpoint["stale_epochs"])
        history = checkpoint["history"]
        if stale_epochs >= args.patience:
            start_epoch = args.epochs
            print("RESUME checkpoint already reached early stopping", flush=True)
    train_manifest = manifest.loc[train_indices].copy()
    started = time.time()
    for epoch in range(start_epoch, args.epochs):
        schedule = build_epoch_schedule(
            train_manifest,
            HierarchicalSamplerConfig(
                batch_size=32, positives_per_batch=8,
                batches_per_epoch=args.batches_per_epoch, seed=seed,
            ),
            epoch=epoch,
        )
        batches = [
            group.sort_values("slot").manifest_index.astype(int).tolist()
            for _, group in schedule.groupby("batch_id", sort=True)
        ]
        loader = DataLoader(
            data, batch_sampler=batches, num_workers=0, pin_memory=True
        )
        model.train()
        losses = []
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            y = batch["label"].to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(
                    batch["ecg"].to(device, non_blocking=True),
                    batch["hrv"].to(device, non_blocking=True),
                )
                loss = F.binary_cross_entropy_with_logits(logits, y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
        validation, y_val, p_val, validation_manifest_indices = evaluate(
            model, data, validation_indices, device
        )
        operating = select_threshold(y_val, p_val)
        at_half = classification(y_val, p_val, 0.5)
        row = {
            "epoch": epoch + 1,
            "train_loss": float(np.mean(losses)),
            "validation": validation,
            "operating_point": operating,
            "at_0_5": at_half,
            "elapsed_minutes": (time.time() - started) / 60.0,
        }
        history.append(row)
        improved = validation["pr_auc"] > best_pr_auc + 1e-5
        if improved:
            best_pr_auc = validation["pr_auc"]
            stale_epochs = 0
        else:
            stale_epochs += 1
        payload = {
            "epoch": epoch,
            "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "hrv_preprocessor": preprocessor.state_dict(),
            "best_pr_auc": best_pr_auc,
            "stale_epochs": stale_epochs,
            "history": history,
            "manifest_sha256": protocol_summary["manifest_sha256"],
            "splits_sha256": protocol_summary["splits_sha256"],
            "fold": args.fold,
        }
        save_checkpoint(last_path, payload)
        if improved:
            save_checkpoint(best_path, payload)
        history_path.write_text(json.dumps(history, indent=2))
        print(
            f"EPOCH {epoch + 1:02d}: train_loss={row['train_loss']:.4f}; "
            f"val_loss={validation['loss']:.4f}; val_pr_auc={validation['pr_auc']:.4f}; "
            f"val_auroc={validation['auroc']:.4f}; threshold={operating['threshold']:.4f}; "
            f"recall={operating['recall']:.3f}; specificity={operating['specificity']:.3f}; "
            f"stale={stale_epochs}/{args.patience}",
            flush=True,
        )
        if stale_epochs >= args.patience:
            print("EARLY STOP", flush=True)
            break
    best = torch.load(best_path, map_location="cpu", weights_only=False)
    model.load_state_dict(best["model_state_dict"])
    model.to(device)
    final_validation, y_val, p_val, validation_manifest_indices = evaluate(
        model, data, validation_indices, device
    )
    operating = select_threshold(y_val, p_val)
    at_half = classification(y_val, p_val, 0.5)
    predictions = manifest.loc[validation_manifest_indices, [
        "manifest_index", "window_id", "subject_id", "source_family", "label",
        "lead_bin", "negative_stratum",
    ]].copy()
    predictions["probability"] = p_val
    predictions.to_csv(args.output_dir / "best_validation_predictions.csv", index=False)
    by_source = {}
    for source, group in predictions.groupby("source_family", sort=True):
        source_labels = group.label.to_numpy(dtype=np.uint8)
        source_probabilities = group.probability.to_numpy(dtype=np.float64)
        by_source[str(source)] = {
            "windows": int(len(group)),
            "positives": int(source_labels.sum()),
            "pr_auc": float(average_precision_score(source_labels, source_probabilities)),
            "auroc": float(roc_auc_score(source_labels, source_probabilities)),
            "operating_point": classification(
                source_labels, source_probabilities, operating["threshold"]
            ),
        }
    recall_by_lead_bin = {}
    positive_predictions = predictions[predictions.label.eq(1)]
    for lead_bin, group in positive_predictions.groupby("lead_bin", sort=True):
        recall_by_lead_bin[str(lead_bin)] = {
            "positives": int(len(group)),
            "true_positives": int(
                (group.probability.to_numpy() >= operating["threshold"]).sum()
            ),
            "recall": float(
                (group.probability.to_numpy() >= operating["threshold"]).mean()
            ),
        }
    summary = {
        "format": "afib_unified_validation_pilot_v1",
        "fold": args.fold,
        "model": "proposed",
        "parameter_count": model.parameter_count,
        "device": torch.cuda.get_device_name(0),
        "recipe": {
            "epochs_max": args.epochs, "patience": args.patience,
            "batches_per_epoch": args.batches_per_epoch, "batch_size": 32,
            "positive_per_batch": 8, "negative_per_batch": 24,
            "loss": "binary_cross_entropy", "mixed_precision": True,
            "learning_rate": 5e-4, "weight_decay": 1e-4,
            "selection_metric": "validation_pr_auc",
            "threshold_rule": "maximize validation recall subject to specificity >= 0.90",
        },
        "train_windows": int(len(train_indices)),
        "validation_windows": int(len(validation_indices)),
        "outer_test_evaluated": False,
        "best_epoch": int(best["epoch"]) + 1,
        "validation_ranking": final_validation,
        "validation_operating_point": operating,
        "validation_at_0_5": at_half,
        "validation_by_source": by_source,
        "validation_positive_recall_by_lead_bin": recall_by_lead_bin,
        "manifest_sha256": protocol_summary["manifest_sha256"],
        "splits_sha256": protocol_summary["splits_sha256"],
        "history": history,
        "gates": {
            "outer_test_not_evaluated": True,
            "patient_disjoint": not train_subjects & validation_subjects
            and not (train_subjects | validation_subjects) & test_subjects,
            "finite_validation_metrics": bool(
                all(np.isfinite(final_validation[key]) for key in ("loss", "pr_auc", "auroc", "brier"))
            ),
            "checkpoint_saved": best_path.is_file(),
            "validation_predictions_saved": (args.output_dir / "best_validation_predictions.csv").is_file(),
        },
    }
    summary["all_gates_pass"] = all(summary["gates"].values())
    (args.output_dir / "validation_pilot_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True)
    )
    print(json.dumps({k: v for k, v in summary.items() if k != "history"}, indent=2), flush=True)
    if not summary["all_gates_pass"]:
        raise RuntimeError("validation pilot gate failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

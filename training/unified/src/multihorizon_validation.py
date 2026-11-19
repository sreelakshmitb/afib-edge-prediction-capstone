"""Fold-0 validation-only training for the five-horizon survival reconstruction."""

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
from torch import nn
from torch.utils.data import DataLoader, Subset

from afib_unified.dataset import UnifiedCacheDataset
from afib_unified.hrv_preprocess import HRVFoldPreprocessor
from afib_unified.manifest import sha256_file
from afib_unified.models import MAX_TRAINABLE_PARAMETERS, build_model
from afib_unified.multihorizon import (
    HORIZONS_MINUTES,
    balanced_discrete_time_nll,
    cumulative_risk_from_logits,
)
from afib_unified.sampling import HierarchicalSamplerConfig, build_epoch_schedule
from afib_unified.validation_metrics import ranking, select_threshold, summarize_predictions


def save_checkpoint(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def rng_state() -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all(),
    }


def restore_rng_state(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    torch.cuda.set_rng_state_all(state["cuda"])


def evaluate(model, data, indices, device, batch_size: int = 32):
    loader = DataLoader(
        Subset(data, [int(index) for index in indices]),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    model.eval()
    event_bins, risks, hazards, manifest_indices = [], [], [], []
    losses = []
    with torch.no_grad():
        for batch in loader:
            ecg = batch["ecg"].to(device, non_blocking=True)
            hrv = batch["hrv"].to(device, non_blocking=True)
            bins = batch["event_bin"].to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(ecg, hrv)
            logits = logits.float()
            loss = balanced_discrete_time_nll(logits, bins)
            losses.append((float(loss), len(bins)))
            hazards.append(torch.sigmoid(logits).cpu().numpy())
            risks.append(cumulative_risk_from_logits(logits).cpu().numpy())
            event_bins.append(bins.cpu().numpy())
            manifest_indices.extend(int(value) for value in batch["manifest_index"])
    return {
        "loss": sum(value * count for value, count in losses) / sum(count for _, count in losses),
        "event_bin": np.concatenate(event_bins).astype(np.int8),
        "risk": np.concatenate(risks).astype(np.float64),
        "hazard": np.concatenate(hazards).astype(np.float64),
        "manifest_index": np.asarray(manifest_indices, dtype=np.int64),
    }


def horizon_diagnostics(predictions: pd.DataFrame) -> dict:
    output = {}
    for horizon in HORIZONS_MINUTES:
        suffix = f"{horizon:02d}"
        labels = predictions[f"target_{suffix}"].to_numpy(dtype=np.uint8)
        probabilities = predictions[f"risk_{suffix}"].to_numpy(dtype=np.float64)
        source_results = {}
        for source, group in predictions.groupby("source_family", sort=True):
            source_results[str(source)] = {
                "ranking": ranking(
                    group[f"target_{suffix}"].to_numpy(dtype=np.uint8),
                    group[f"risk_{suffix}"].to_numpy(dtype=np.float64),
                ),
                "operating_point": select_threshold(
                    group[f"target_{suffix}"].to_numpy(dtype=np.uint8),
                    group[f"risk_{suffix}"].to_numpy(dtype=np.float64),
                ),
            }
        output[suffix] = {
            "ranking": ranking(labels, probabilities),
            "operating_point": select_threshold(labels, probabilities),
            "by_source": source_results,
        }
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy-cache", type=Path, required=True)
    parser.add_argument("--iridia-cache", type=Path, required=True)
    parser.add_argument("--protocol-dir", type=Path, required=True)
    parser.add_argument("--target-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batches-per-epoch", type=int, default=50)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.fold != 0:
        raise ValueError("this structural validation experiment is predeclared for fold 0 only")

    seed = 7
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cached_path = args.protocol_dir / "unified_cached_manifest.csv"
    splits_path = args.protocol_dir / "locked_patient_splits.json"
    protocol = json.loads((args.protocol_dir / "protocol_summary.json").read_text())
    target_path = args.target_dir / "multihorizon_cached_manifest.csv"
    target_summary = json.loads((args.target_dir / "target_summary.json").read_text())
    cached_hash = sha256_file(cached_path)
    splits_hash = sha256_file(splits_path)
    target_hash = sha256_file(target_path)
    if cached_hash != protocol["manifest_sha256"]:
        raise ValueError("cached manifest hash mismatch")
    if splits_hash != protocol["splits_sha256"]:
        raise ValueError("frozen split hash mismatch")
    if target_summary["cached_manifest_sha256"] != cached_hash:
        raise ValueError("targets were built from a different cached manifest")
    if target_summary["multihorizon_manifest_sha256"] != target_hash:
        raise ValueError("multi-horizon target hash mismatch")
    if not target_summary.get("all_gates_pass"):
        raise ValueError("multi-horizon target gates did not pass")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "multihorizon_summary.json"
    if args.resume and summary_path.is_file():
        complete = json.loads(summary_path.read_text())
        if (
            complete.get("all_gates_pass")
            and complete.get("target_manifest_sha256") == target_hash
            and complete.get("splits_sha256") == splits_hash
        ):
            print("MULTI-HORIZON VALIDATION ALREADY COMPLETE")
            print(json.dumps(complete, indent=2))
            return 0

    manifest = pd.read_csv(target_path).sort_values("manifest_index").reset_index(drop=True)
    split_data = json.loads(splits_path.read_text())
    fold = next(row for row in split_data["folds"] if int(row["fold"]) == args.fold)
    train_subjects = set(fold["train_subjects"])
    validation_subjects = set(fold["validation_subjects"])
    test_subjects = set(fold["test_subjects"])
    if train_subjects & validation_subjects or (train_subjects | validation_subjects) & test_subjects:
        raise AssertionError("patient leakage in frozen split")
    train_indices = manifest.index[manifest.subject_id.astype(str).isin(train_subjects)].to_numpy()
    validation_indices = manifest.index[
        manifest.subject_id.astype(str).isin(validation_subjects)
    ].to_numpy()
    # Outer-test indices are deliberately never constructed.

    raw = UnifiedCacheDataset(
        args.legacy_cache, args.iridia_cache, target_path, None, "none"
    )
    last_path = args.output_dir / "last_checkpoint.pt"
    best_path = args.output_dir / "best_checkpoint.pt"
    history_path = args.output_dir / "history.json"
    checkpoint = None
    if args.resume and last_path.is_file():
        checkpoint = torch.load(last_path, map_location="cpu", weights_only=False)
        if checkpoint["target_manifest_sha256"] != target_hash:
            raise ValueError("resume checkpoint belongs to different targets")
        if checkpoint["splits_sha256"] != splits_hash:
            raise ValueError("resume checkpoint belongs to different splits")
        preprocessor = HRVFoldPreprocessor.from_state_dict(checkpoint["hrv_preprocessor"])
        print(f"RESUME multi-horizon training from epoch {checkpoint['epoch'] + 2}", flush=True)
    else:
        print(f"Fit HRV preprocessing on {len(train_indices)} training windows only", flush=True)
        preprocessor = HRVFoldPreprocessor().fit(raw.raw_hrv_for_indices(train_indices))
    data = UnifiedCacheDataset(
        args.legacy_cache, args.iridia_cache, target_path, preprocessor, "segment_zscore"
    )
    device = torch.device("cuda")
    model = build_model("proposed-groupnorm-multihorizon").to(device)
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
        restore_rng_state(checkpoint["rng_state"])
        if stale_epochs >= args.patience and best_path.is_file():
            start_epoch = args.epochs
            print("RESUME checkpoint already reached early stopping", flush=True)

    train_manifest = manifest.loc[train_indices].copy()
    started = time.time()
    for epoch in range(start_epoch, args.epochs):
        schedule = build_epoch_schedule(
            train_manifest,
            HierarchicalSamplerConfig(
                batch_size=32,
                positives_per_batch=8,
                batches_per_epoch=args.batches_per_epoch,
                positive_lead_bins=("00_04", "04_08", "08_12", "12_16", "16_20"),
                seed=seed,
            ),
            epoch=epoch,
        )
        batches = [
            group.sort_values("slot").manifest_index.astype(int).tolist()
            for _, group in schedule.groupby("batch_id", sort=True)
        ]
        loader = DataLoader(data, batch_sampler=batches, num_workers=0, pin_memory=True)
        model.train()
        losses = []
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            ecg = batch["ecg"].to(device, non_blocking=True)
            hrv = batch["hrv"].to(device, non_blocking=True)
            bins = batch["event_bin"].to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(ecg, hrv)
            loss = balanced_discrete_time_nll(logits.float(), bins)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
        validation = evaluate(model, data, validation_indices, device)
        y20 = manifest.loc[validation["manifest_index"], "target_20"].to_numpy(dtype=np.uint8)
        p20 = validation["risk"][:, -1]
        metrics20 = ranking(y20, p20)
        operating20 = select_threshold(y20, p20)
        row = {
            "epoch": epoch + 1,
            "train_loss": float(np.mean(losses)),
            "validation_loss": validation["loss"],
            "validation_20min_pr_auc": metrics20["pr_auc"],
            "validation_20min_auroc": metrics20["auroc"],
            "validation_20min_recall_at_90_spec": operating20["recall"],
            "elapsed_minutes": (time.time() - started) / 60.0,
        }
        history.append(row)
        improved = metrics20["pr_auc"] > best_pr_auc + 1e-5
        if improved:
            best_pr_auc = metrics20["pr_auc"]
            stale_epochs = 0
        else:
            stale_epochs += 1
        payload = {
            "epoch": epoch,
            "model_state_dict": {
                key: value.detach().cpu() for key, value in model.state_dict().items()
            },
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "hrv_preprocessor": preprocessor.state_dict(),
            "best_pr_auc": best_pr_auc,
            "stale_epochs": stale_epochs,
            "history": history,
            "target_manifest_sha256": target_hash,
            "splits_sha256": splits_hash,
            "fold": args.fold,
            "rng_state": rng_state(),
        }
        save_checkpoint(last_path, payload)
        if improved:
            save_checkpoint(best_path, payload)
        history_path.write_text(json.dumps(history, indent=2))
        print(
            f"EPOCH {epoch + 1:02d}: train_loss={row['train_loss']:.4f}; "
            f"val_loss={validation['loss']:.4f}; PR-AUC20={metrics20['pr_auc']:.4f}; "
            f"AUROC20={metrics20['auroc']:.4f}; recall20@90spec={operating20['recall']:.3f}; "
            f"stale={stale_epochs}/{args.patience}",
            flush=True,
        )
        if stale_epochs >= args.patience:
            print("EARLY STOP", flush=True)
            break

    if not best_path.is_file():
        raise RuntimeError("no best checkpoint was saved")
    best = torch.load(best_path, map_location="cpu", weights_only=False)
    model.load_state_dict(best["model_state_dict"])
    model.to(device)
    validation = evaluate(model, data, validation_indices, device)
    metadata_columns = [
        "manifest_index", "window_id", "subject_id", "source_family", "label",
        "event_id", "lead_bin", "negative_stratum", "record_id", "event_bin",
        "lead_time_minutes", "target_04", "target_08", "target_12", "target_16",
        "target_20",
    ]
    predictions = manifest.loc[validation["manifest_index"], metadata_columns].copy()
    for position, horizon in enumerate(HORIZONS_MINUTES):
        suffix = f"{horizon:02d}"
        predictions[f"hazard_{suffix}"] = validation["hazard"][:, position]
        predictions[f"risk_{suffix}"] = validation["risk"][:, position]
    prediction_path = args.output_dir / "best_validation_predictions.csv"
    predictions.to_csv(prediction_path, index=False)
    per_horizon = horizon_diagnostics(predictions)
    twenty = predictions[
        [
            "window_id", "subject_id", "source_family", "label", "event_id",
            "lead_bin", "negative_stratum", "risk_20",
        ]
    ].rename(columns={"risk_20": "probability"})
    twenty_diagnostics = summarize_predictions(twenty)
    monotonic_risk = bool((np.diff(validation["risk"], axis=1) >= -1e-7).all())
    gates = {
        "outer_test_not_evaluated": True,
        "patient_disjoint": not train_subjects & validation_subjects
        and not (train_subjects | validation_subjects) & test_subjects,
        "target_hash_locked": target_hash == target_summary["multihorizon_manifest_sha256"],
        "groupnorm_installed": not any(isinstance(module, nn.BatchNorm1d) for module in model.modules())
        and any(isinstance(module, nn.GroupNorm) for module in model.modules()),
        "parameter_budget": model.parameter_count < MAX_TRAINABLE_PARAMETERS,
        "five_hazards_emitted": validation["hazard"].shape[1] == 5,
        "cumulative_risk_monotonic": monotonic_risk,
        "finite_validation_predictions": bool(np.isfinite(validation["risk"]).all()),
        "checkpoint_saved": best_path.is_file(),
        "validation_predictions_saved": prediction_path.is_file(),
    }
    summary = {
        "format": "afib_multihorizon_validation_v1",
        "fold": args.fold,
        "model": "proposed-groupnorm-multihorizon",
        "parameter_count": model.parameter_count,
        "device": torch.cuda.get_device_name(0),
        "recipe": {
            "horizons_minutes": list(HORIZONS_MINUTES),
            "loss": "class-balanced per-window discrete-time survival NLL",
            "batch_size": 32,
            "positive_per_batch": 8,
            "negative_per_batch": 24,
            "positive_sampling": "source, patient, event, and lead-bin balanced",
            "negative_sampling": "source and proximity-stratum balanced",
            "learning_rate": 5e-4,
            "weight_decay": 1e-4,
            "selection_metric": "validation_20min_pr_auc",
            "specificity_floor": 0.90,
        },
        "train_windows": int(len(train_indices)),
        "validation_windows": int(len(validation_indices)),
        "best_epoch": int(best["epoch"]) + 1,
        "per_horizon": per_horizon,
        "twenty_minute_diagnostics": twenty_diagnostics,
        "outer_test_evaluated": False,
        "target_manifest_sha256": target_hash,
        "splits_sha256": splits_hash,
        "history": history,
        "gates": gates,
        "all_gates_pass": all(gates.values()),
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    printable = {key: value for key, value in summary.items() if key != "history"}
    print(json.dumps(printable, indent=2), flush=True)
    if not summary["all_gates_pass"]:
        raise RuntimeError("multi-horizon validation integrity gate failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

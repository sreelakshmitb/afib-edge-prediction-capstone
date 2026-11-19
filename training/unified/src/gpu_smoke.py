"""One exact balanced GPU batch through the frozen unified protocol."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from afib_unified.dataset import UnifiedCacheDataset
from afib_unified.hrv_preprocess import HRVFoldPreprocessor
from afib_unified.manifest import sha256_file
from afib_unified.models import MAX_TRAINABLE_PARAMETERS, build_model
from afib_unified.sampling import HierarchicalSamplerConfig, build_epoch_schedule


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy-cache", type=Path, required=True)
    parser.add_argument("--iridia-cache", type=Path, required=True)
    parser.add_argument("--protocol-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fold", type=int, default=0)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the GPU smoke test")
    manifest_path = args.protocol_dir / "unified_cached_manifest.csv"
    splits_path = args.protocol_dir / "locked_patient_splits.json"
    manifest = pd.read_csv(manifest_path).sort_values("manifest_index").reset_index(drop=True)
    split_data = json.loads(splits_path.read_text())
    fold_data = next(row for row in split_data["folds"] if int(row["fold"]) == args.fold)
    train_subjects = set(fold_data["train_subjects"])
    validation_subjects = set(fold_data["validation_subjects"])
    test_subjects = set(fold_data["test_subjects"])
    if train_subjects & validation_subjects or (train_subjects | validation_subjects) & test_subjects:
        raise AssertionError("patient leakage in frozen split")
    train_indices = manifest.index[manifest.subject_id.astype(str).isin(train_subjects)].to_numpy()
    raw = UnifiedCacheDataset(
        args.legacy_cache, args.iridia_cache, manifest_path,
        hrv_preprocessor=None, ecg_normalization="none",
    )
    print(f"Fitting HRV preprocessing on {len(train_indices)} inner-training windows only...", flush=True)
    preprocessor = HRVFoldPreprocessor().fit(raw.raw_hrv_for_indices(train_indices))
    data = UnifiedCacheDataset(
        args.legacy_cache, args.iridia_cache, manifest_path,
        hrv_preprocessor=preprocessor, ecg_normalization="segment_zscore",
    )
    train_manifest = manifest.loc[train_indices].copy()
    schedule = build_epoch_schedule(
        train_manifest,
        HierarchicalSamplerConfig(
            batch_size=32, positives_per_batch=8, batches_per_epoch=1, seed=7
        ),
        epoch=0,
    ).sort_values(["batch_id", "slot"])
    batch_indices = schedule.manifest_index.to_numpy(dtype=np.int64)
    samples = [data[int(index)] for index in batch_indices]
    ecg = torch.stack([item["ecg"] for item in samples])
    hrv = torch.stack([item["hrv"] for item in samples])
    labels = torch.stack([item["label"] for item in samples])
    device = torch.device("cuda")
    model = build_model("proposed").to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    torch.cuda.reset_peak_memory_stats(device)
    optimizer.zero_grad(set_to_none=True)
    logits = model(ecg.to(device), hrv.to(device))
    loss = F.binary_cross_entropy_with_logits(logits, labels.to(device))
    loss.backward()
    gradient_norm = float(
        torch.sqrt(
            sum(
                torch.sum(parameter.grad.detach() ** 2)
                for parameter in model.parameters()
                if parameter.grad is not None
            )
        ).item()
    )
    optimizer.step()
    torch.cuda.synchronize(device)
    source_by_label = {
        f"label_{int(label)}:{family}": int(count)
        for (label, family), count in schedule.groupby(["label", "source_family"]).size().items()
    }
    gates = {
        "cuda_used": logits.is_cuda,
        "batch_shape_ecg": tuple(ecg.shape) == (32, 20, 1, 7500),
        "batch_shape_hrv": tuple(hrv.shape) == (32, 20, 6),
        "exact_8_positive_24_negative": int(labels.sum()) == 8 and len(labels) == 32,
        "source_balanced_within_label": source_by_label == {
            "label_0:iridia": 12, "label_0:legacy": 12,
            "label_1:iridia": 4, "label_1:legacy": 4,
        },
        "finite_inputs": bool(torch.isfinite(ecg).all() and torch.isfinite(hrv).all()),
        "finite_logits": bool(torch.isfinite(logits).all()),
        "finite_loss": bool(torch.isfinite(loss)),
        "finite_nonzero_gradient": bool(np.isfinite(gradient_norm) and gradient_norm > 0),
        "parameter_budget": model.parameter_count < MAX_TRAINABLE_PARAMETERS,
        "patient_disjoint": not train_subjects & validation_subjects
        and not (train_subjects | validation_subjects) & test_subjects,
    }
    summary = {
        "format": "afib_unified_gpu_smoke_v1",
        "fold": args.fold,
        "device": torch.cuda.get_device_name(0),
        "parameter_count": model.parameter_count,
        "loss": float(loss.detach().cpu()),
        "gradient_norm": gradient_norm,
        "peak_gpu_memory_mib": float(torch.cuda.max_memory_allocated(device) / 1024**2),
        "train_windows_used_for_hrv_fit": int(len(train_indices)),
        "batch_positive_count": int(labels.sum()),
        "source_by_label": source_by_label,
        "positive_lead_bins": {
            str(k): int(v)
            for k, v in schedule.loc[schedule.label.eq(1), "lead_bin"].value_counts().sort_index().items()
        },
        "manifest_sha256": sha256_file(manifest_path),
        "splits_sha256": sha256_file(splits_path),
        "hrv_preprocessor": preprocessor.state_dict(),
        "gates": gates,
        "all_gates_pass": all(gates.values()),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "gpu_smoke_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True)
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    if not summary["all_gates_pass"]:
        raise RuntimeError("GPU smoke gate failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

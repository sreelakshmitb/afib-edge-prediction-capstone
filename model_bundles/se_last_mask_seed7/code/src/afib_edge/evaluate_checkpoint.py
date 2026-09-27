"""Evaluate a saved fold checkpoint with validation-only threshold selection."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from .dataset import CacheWindowDataset
from .hrv_preprocess import HRVFoldPreprocessor
from .models import build_model
from .training import (
    BinaryFocalLoss,
    make_outer_fold_indices,
    make_validation_split,
    metrics_from_predictions,
    predict_loader,
    resolve_device,
    select_f1_threshold,
)


def _load_checkpoint(path: Path) -> dict[str, object]:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # PyTorch versions before weights_only was available.
        checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise ValueError("checkpoint must contain a dictionary")
    return checkpoint


def _write_predictions(
    path: Path,
    split: str,
    dataset: CacheWindowDataset,
    indices: np.ndarray,
    predictions: dict[str, object],
    threshold: float,
) -> None:
    labels = np.asarray(predictions["labels"], dtype=np.uint8)
    probabilities = np.asarray(predictions["probabilities"], dtype=np.float64)
    if len(indices) != len(labels):
        raise ValueError("prediction count does not match split indices")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "split",
                "dataset_index",
                "subject_id",
                "start_sample_250hz",
                "label",
                "probability",
                "prediction",
                "threshold",
            ),
        )
        writer.writeheader()
        for dataset_index, label, probability in zip(indices, labels, probabilities):
            ref = dataset.window_refs[int(dataset_index)]
            writer.writerow(
                {
                    "split": split,
                    "dataset_index": int(dataset_index),
                    "subject_id": ref.subject_id,
                    "start_sample_250hz": ref.start_sample_250hz,
                    "label": int(label),
                    "probability": float(probability),
                    "prediction": int(probability >= threshold),
                    "threshold": threshold,
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--validation-fold", type=int, default=0)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    checkpoint = _load_checkpoint(args.checkpoint)
    model_name = str(checkpoint["model_name"])
    fold = int(checkpoint["fold"])
    device = resolve_device(args.device)

    checkpoint_summary = checkpoint.get("summary", {})
    checkpoint_config = checkpoint_summary.get("training_config", {})
    if not isinstance(checkpoint_config, dict):
        checkpoint_config = {}
    dataset = CacheWindowDataset(
        args.cache_root,
        args.manifest,
        ecg_normalization=str(checkpoint_config.get("ecg_normalization", "none")),
    )
    outer_train, outer_test = make_outer_fold_indices(
        dataset.labels, dataset.subject_ids, n_splits=args.n_splits, fold=fold
    )
    _, validation = make_validation_split(
        outer_train,
        dataset.subject_ids,
        n_splits=args.n_splits,
        validation_fold=args.validation_fold,
    )
    dataset.hrv_preprocessor = HRVFoldPreprocessor.from_state_dict(
        checkpoint["hrv_preprocessor"]
    )
    loader_kwargs = {
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
    }
    validation_loader = DataLoader(Subset(dataset, validation.tolist()), **loader_kwargs)
    test_loader = DataLoader(Subset(dataset, outer_test.tolist()), **loader_kwargs)

    model = build_model(model_name).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    criterion = (
        BinaryFocalLoss(gamma=float(checkpoint_config.get("focal_gamma", 2.0)))
        if str(checkpoint_config.get("loss", "focal")) == "focal"
        else torch.nn.BCEWithLogitsLoss()
    )
    validation_predictions = predict_loader(model, validation_loader, criterion, device)
    threshold_selection = select_f1_threshold(
        np.asarray(validation_predictions["labels"]),
        np.asarray(validation_predictions["probabilities"]),
    )
    threshold = threshold_selection["threshold"]
    test_predictions = predict_loader(model, test_loader, criterion, device)

    summary = {
        "model": model_name,
        "fold": fold,
        "device": str(device),
        "checkpoint": str(args.checkpoint),
        "checkpoint_best_epoch": checkpoint_summary.get("best_epoch"),
        "checkpoint_selection_metric": checkpoint_summary.get(
            "selection_metric", "legacy_validation_f1_at_0.5"
        ),
        "threshold_source": "inner_validation_f1_max",
        "selected_threshold": threshold,
        "validation_metrics": metrics_from_predictions(
            validation_predictions, threshold=threshold
        ),
        "test_metrics": metrics_from_predictions(test_predictions, threshold=threshold),
        "test_metrics_at_0_5": metrics_from_predictions(test_predictions, threshold=0.5),
        "validation_windows": len(validation),
        "outer_test_windows": len(outer_test),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{model_name}_fold{fold}"
    _write_predictions(
        args.output_dir / f"{stem}_validation_predictions.csv",
        "validation",
        dataset,
        validation,
        validation_predictions,
        threshold,
    )
    _write_predictions(
        args.output_dir / f"{stem}_test_predictions.csv",
        "outer_test",
        dataset,
        outer_test,
        test_predictions,
        threshold,
    )
    summary_path = args.output_dir / f"{stem}_evaluation.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

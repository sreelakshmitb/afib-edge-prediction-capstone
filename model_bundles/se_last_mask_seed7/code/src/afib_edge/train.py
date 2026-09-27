"""Command-line entry point for a one-fold training smoke test."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch

from .dataset import CacheWindowDataset
from .training import TrainingConfig, train_one_fold


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--model", default="proposed", choices=("paper", "proposed"))
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-eval-batches", type=int)
    parser.add_argument(
        "--ecg-normalization", default="none", choices=("none", "segment_zscore")
    )
    parser.add_argument("--sampling", default="balanced", choices=("balanced", "natural"))
    parser.add_argument("--loss", default="focal", choices=("focal", "bce"))
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    config = TrainingConfig(
        model_name=args.model,
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.patience,
        n_splits=args.n_splits,
        device=args.device,
        num_workers=args.num_workers,
        max_train_batches=args.max_train_batches,
        max_eval_batches=args.max_eval_batches,
        ecg_normalization=args.ecg_normalization,
        sampling=args.sampling,
        loss=args.loss,
    )
    dataset = CacheWindowDataset(args.cache_root, args.manifest)
    result = train_one_fold(dataset, config, fold=args.fold)
    model = result["model"]
    summary = {
        "model": args.model,
        "fold": args.fold,
        "training_config": asdict(config),
        "device": result["device"],
        "parameter_count": result["parameter_count"],
        "best_epoch": result["best_epoch"],
        "selection_metric": result["selection_metric"],
        "selected_threshold": result["selected_threshold"],
        "dataset_windows": len(dataset),
        "outer_train_windows": len(result["outer_train_indices"]),
        "validation_windows": len(result["validation_indices"]),
        "outer_test_windows": len(result["outer_test_indices"]),
        "test_metrics": result["test_metrics"],
        "test_metrics_at_0_5": result["test_metrics_at_0_5"],
        "validation_metrics": result["validation_metrics"],
        "history": result["history"],
    }

    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = args.output_dir / f"{args.model}_fold{args.fold}.pt"
        torch.save(
            {
                "model_name": args.model,
                "fold": args.fold,
                "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "hrv_preprocessor": result["hrv_preprocessor"].state_dict(),
                "summary": summary,
            },
            checkpoint_path,
        )
        summary["checkpoint"] = str(checkpoint_path)
        (args.output_dir / f"{args.model}_fold{args.fold}_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

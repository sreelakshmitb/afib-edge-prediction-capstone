"""Resumable five-fold evaluation for a locked training recipe."""

from __future__ import annotations

import argparse
import gc
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from .dataset import CacheWindowDataset
from .training import TrainingConfig, train_one_fold


def _save_checkpoint(path: Path, model: torch.nn.Module, result: dict[str, object], summary: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "model_name": summary["model"],
            "fold": summary["fold"],
            "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "hrv_preprocessor": result["hrv_preprocessor"].state_dict(),
            "summary": summary,
        },
        temporary,
    )
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--models", nargs="+", choices=("paper", "proposed"), default=["paper", "proposed"])
    parser.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--ecg-normalization", default="segment_zscore", choices=("none", "segment_zscore"))
    parser.add_argument("--sampling", default="natural", choices=("balanced", "natural"))
    parser.add_argument("--loss", default="bce", choices=("focal", "bce"))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.epochs <= 0 or args.patience < 0:
        parser.error("epochs must be positive and patience cannot be negative")
    if len(set(args.folds)) != len(args.folds) or any(fold < 0 or fold >= 5 for fold in args.folds):
        parser.error("folds must be unique integers from 0 through 4")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    all_results: list[dict[str, object]] = []

    for model_name in args.models:
        for fold in args.folds:
            fold_dir = args.output_dir / model_name / f"fold{fold}"
            summary_path = fold_dir / f"{model_name}_fold{fold}_summary.json"
            checkpoint_path = fold_dir / f"{model_name}_fold{fold}.pt"
            if args.resume and summary_path.is_file() and checkpoint_path.is_file():
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                all_results.append(summary)
                print(f"RESUME skip {model_name} fold {fold}: existing summary", flush=True)
                continue

            fold_started = time.monotonic()
            fold_dir.mkdir(parents=True, exist_ok=True)
            print(
                f"START {model_name} fold {fold} "
                f"(normalization={args.ecg_normalization}, sampling={args.sampling}, loss={args.loss})",
                flush=True,
            )
            dataset = CacheWindowDataset(
                args.cache_root,
                args.manifest,
                ecg_normalization=args.ecg_normalization,
            )
            config = TrainingConfig(
                model_name=model_name,
                batch_size=args.batch_size,
                epochs=args.epochs,
                patience=args.patience,
                device=args.device,
                seed=7,
                ecg_normalization=args.ecg_normalization,
                sampling=args.sampling,
                loss=args.loss,
            )
            result = train_one_fold(
                dataset,
                config,
                fold=fold,
                verbose=True,
                evaluate_outer_test=True,
            )
            summary = {
                "model": model_name,
                "fold": fold,
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
                "validation_metrics": result["validation_metrics"],
                "test_metrics": result["test_metrics"],
                "test_metrics_at_0_5": result["test_metrics_at_0_5"],
                "history": result["history"],
                "elapsed_seconds": time.monotonic() - fold_started,
            }
            _save_checkpoint(checkpoint_path, result["model"], result, summary)
            temporary_summary = summary_path.with_suffix(summary_path.suffix + ".tmp")
            temporary_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
            temporary_summary.replace(summary_path)
            all_results.append(summary)
            print(
                f"FINISH {model_name} fold {fold}: "
                f"val_pr_auc={summary['validation_metrics'].get('pr_auc')} "
                f"test_pr_auc={summary['test_metrics'].get('pr_auc')} "
                f"elapsed={summary['elapsed_seconds'] / 60:.1f} min",
                flush=True,
            )
            del result, dataset
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def aggregate(model: str) -> dict[str, object]:
        rows = [row for row in all_results if row["model"] == model]
        if not rows:
            return {"model": model, "folds": 0}
        metrics = ("accuracy", "precision", "recall", "specificity", "f1", "auroc", "pr_auc")
        aggregate_metrics: dict[str, dict[str, float | None]] = {}
        for split in ("validation_metrics", "test_metrics"):
            aggregate_metrics[split] = {}
            for metric in metrics:
                values = [row[split].get(metric) for row in rows if row[split].get(metric) is not None]
                aggregate_metrics[split][f"{metric}_mean"] = float(np.mean(values)) if values else None
                aggregate_metrics[split][f"{metric}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else None
        return {"model": model, "folds": len(rows), "metrics": aggregate_metrics}

    output = {
        "training_config": {
            "ecg_normalization": args.ecg_normalization,
            "sampling": args.sampling,
            "loss": args.loss,
            "epochs": args.epochs,
            "patience": args.patience,
        },
        "selection_rule": "model and recipe are chosen from validation metrics; outer-test metrics are reported after the choice",
        "elapsed_seconds": time.monotonic() - started,
        "folds": all_results,
        "aggregate": [aggregate(model) for model in args.models],
    }
    (args.output_dir / "cross_validation_summary.json").write_text(
        json.dumps(output, indent=2), encoding="utf-8"
    )
    print(json.dumps(output["aggregate"], indent=2), flush=True)


if __name__ == "__main__":
    main()

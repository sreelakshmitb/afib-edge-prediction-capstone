"""Run a validation-only fold-0 recipe pilot.

Each recipe is trained on the inner-training patients and selected by inner
validation PR-AUC. The outer-test windows are never iterated, so this pilot
cannot tune the recipe against the final evaluation fold.
"""

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


RECIPES: tuple[dict[str, str], ...] = (
    {"name": "raw_focal_balanced", "ecg_normalization": "none", "sampling": "balanced", "loss": "focal"},
    {"name": "zscore_focal_balanced", "ecg_normalization": "segment_zscore", "sampling": "balanced", "loss": "focal"},
    {"name": "raw_bce_natural", "ecg_normalization": "none", "sampling": "natural", "loss": "bce"},
    {"name": "zscore_bce_natural", "ecg_normalization": "segment_zscore", "sampling": "natural", "loss": "bce"},
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model", default="proposed", choices=("paper", "proposed"))
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda", choices=("auto", "cpu", "cuda"))
    args = parser.parse_args()
    if args.epochs <= 0 or args.patience < 0:
        parser.error("epochs must be positive and patience cannot be negative")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    results: list[dict[str, object]] = []

    for recipe_index, recipe in enumerate(RECIPES, start=1):
        recipe_start = time.monotonic()
        print(
            f"[{recipe_index}/{len(RECIPES)}] START {recipe['name']} "
            f"(normalization={recipe['ecg_normalization']}, sampling={recipe['sampling']}, loss={recipe['loss']})",
            flush=True,
        )
        dataset = CacheWindowDataset(
            args.cache_root,
            args.manifest,
            ecg_normalization=recipe["ecg_normalization"],
        )
        config = TrainingConfig(
            model_name=args.model,
            batch_size=args.batch_size,
            epochs=args.epochs,
            patience=args.patience,
            device=args.device,
            seed=7,
            ecg_normalization=recipe["ecg_normalization"],
            sampling=recipe["sampling"],
            loss=recipe["loss"],
        )
        result = train_one_fold(
            dataset,
            config,
            fold=args.fold,
            verbose=True,
            evaluate_outer_test=False,
        )
        validation = result["validation_metrics"]
        if not isinstance(validation, dict):
            raise RuntimeError("pilot validation metrics were not produced")
        summary = {
            "recipe": recipe["name"],
            "model": args.model,
            "fold": args.fold,
            "training_config": asdict(config),
            "best_epoch": result["best_epoch"],
            "selection_metric": result["selection_metric"],
            "selected_threshold": result["selected_threshold"],
            "validation_metrics": validation,
            "inner_train_windows": len(result["inner_train_indices"]),
            "validation_windows": len(result["validation_indices"]),
            "outer_test_iterated": False,
            "elapsed_seconds": time.monotonic() - recipe_start,
        }
        results.append(summary)
        (args.output_dir / f"{recipe['name']}.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        print(
            f"[{recipe_index}/{len(RECIPES)}] FINISH {recipe['name']} "
            f"val_pr_auc={validation.get('pr_auc')} "
            f"best_epoch={result['best_epoch']} "
            f"elapsed={(time.monotonic() - recipe_start) / 60:.1f} min",
            flush=True,
        )
        del result, dataset
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    ranking = sorted(
        results,
        key=lambda row: -float(row["validation_metrics"].get("pr_auc") or -np.inf),
    )
    output = {
        "scope": "recipe selection uses inner validation PR-AUC only",
        "outer_test_iterated": False,
        "total_elapsed_seconds": time.monotonic() - started,
        "ranking": ranking,
    }
    (args.output_dir / "pilot_summary.json").write_text(
        json.dumps(output, indent=2), encoding="utf-8"
    )
    print(json.dumps(output, indent=2), flush=True)


if __name__ == "__main__":
    main()

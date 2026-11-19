"""Two predeclared fold-0 validation-only repairs; outer test is never loaded."""

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
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset

from afib_unified.dataset import UnifiedCacheDataset
from afib_unified.hrv_preprocess import HRVFoldPreprocessor
from afib_unified.manifest import sha256_file
from afib_unified.models import MAX_TRAINABLE_PARAMETERS, build_model
from afib_unified.sampling import HierarchicalSamplerConfig, build_epoch_schedule
from afib_unified.validation_metrics import summarize_predictions


RECIPES = (
    "groupnorm",
    "groupnorm-curriculum",
    "groupnorm-curriculum-ecg-only",
    "groupnorm-curriculum-hrv-only",
    "groupnorm-curriculum-domain-align",
    "groupnorm-curriculum-harmonized-negatives",
    "groupnorm-curriculum-corrected-eligibility",
)
ALL_LEAD_BINS = ("00_04", "04_08", "08_12", "12_16", "16_20")


def lead_bins_for_epoch(recipe: str, epoch: int) -> tuple[str, ...]:
    if recipe == "groupnorm":
        return ALL_LEAD_BINS
    if recipe not in RECIPES:
        raise ValueError(f"unknown recipe: {recipe}")
    if epoch < 3:
        return ALL_LEAD_BINS[:2]
    if epoch < 6:
        return ALL_LEAD_BINS[:3]
    return ALL_LEAD_BINS


def input_mode_for_recipe(recipe: str) -> str:
    if recipe.endswith("-ecg-only"):
        return "ecg-only"
    if recipe.endswith("-hrv-only"):
        return "hrv-only"
    return "full"


def apply_input_mode(ecg, hrv, input_mode: str):
    if input_mode == "full":
        return ecg, hrv
    if input_mode == "ecg-only":
        return ecg, torch.zeros_like(hrv)
    if input_mode == "hrv-only":
        return torch.zeros_like(ecg), hrv
    raise ValueError(f"unknown input mode: {input_mode}")


def conditional_domain_mean_loss(representation, labels, source_is_iridia):
    """Align source means separately inside the AF-positive and negative classes."""
    features = representation.float()
    center = features.detach().mean(dim=0, keepdim=True)
    scale = features.detach().std(dim=0, unbiased=False, keepdim=True).clamp_min(0.05)
    normalized = (features - center) / scale
    terms = []
    for label in (0, 1):
        legacy = normalized[(labels == label) & ~source_is_iridia]
        iridia = normalized[(labels == label) & source_is_iridia]
        if len(legacy) < 2 or len(iridia) < 2:
            raise ValueError(
                f"conditional alignment needs at least two samples per source for label {label}"
            )
        terms.append(F.mse_loss(legacy.mean(dim=0), iridia.mean(dim=0)))
    return torch.stack(terms).mean()


def evaluate(model, data, indices, device, input_mode="full", batch_size=32):
    loader = DataLoader(
        Subset(data, [int(index) for index in indices]),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    model.eval()
    labels, probabilities, manifest_indices = [], [], []
    total_loss = 0.0
    with torch.no_grad():
        for batch in loader:
            y = batch["label"].to(device, non_blocking=True)
            ecg, hrv = apply_input_mode(
                batch["ecg"].to(device, non_blocking=True),
                batch["hrv"].to(device, non_blocking=True),
                input_mode,
            )
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(ecg, hrv)
                loss = F.binary_cross_entropy_with_logits(logits, y)
            total_loss += float(loss) * len(y)
            labels.append(y.cpu().numpy())
            probabilities.append(torch.sigmoid(logits).float().cpu().numpy())
            manifest_indices.extend(int(value) for value in batch["manifest_index"])
    y = np.concatenate(labels).astype(np.uint8)
    p = np.concatenate(probabilities).astype(np.float64)
    return {
        "loss": total_loss / len(y),
        "pr_auc": float(average_precision_score(y, p)),
        "auroc": float(roc_auc_score(y, p)),
    }, y, p, np.asarray(manifest_indices, dtype=np.int64)


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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy-cache", type=Path, required=True)
    parser.add_argument("--iridia-cache", type=Path, required=True)
    parser.add_argument("--protocol-dir", type=Path, required=True)
    parser.add_argument("--manifest-path", type=Path)
    parser.add_argument("--manifest-summary", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--recipe", choices=RECIPES, required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batches-per-epoch", type=int, default=50)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.fold != 0:
        raise ValueError("controlled repair is predeclared for fold 0 only")
    seed = 7
    input_mode = input_mode_for_recipe(args.recipe)
    use_domain_alignment = args.recipe == "groupnorm-curriculum-domain-align"
    domain_alignment_weight = 0.05
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    manifest_path = args.manifest_path or (
        args.protocol_dir / "unified_cached_manifest.csv"
    )
    splits_path = args.protocol_dir / "locked_patient_splits.json"
    protocol_summary = json.loads((args.protocol_dir / "protocol_summary.json").read_text())
    manifest_hash = sha256_file(manifest_path)
    splits_hash = sha256_file(splits_path)
    harmonized_negatives = args.recipe == "groupnorm-curriculum-harmonized-negatives"
    uses_harmonized_manifest = args.recipe in (
        "groupnorm-curriculum-harmonized-negatives",
        "groupnorm-curriculum-corrected-eligibility",
    )
    if uses_harmonized_manifest:
        if args.manifest_path is None or args.manifest_summary is None:
            raise ValueError("harmonized repair requires --manifest-path and --manifest-summary")
        harmonized_summary = json.loads(args.manifest_summary.read_text())
        if not harmonized_summary.get("all_gates_pass"):
            raise ValueError("harmonized manifest gates did not pass")
        if harmonized_summary["harmonized_manifest_sha256"] != manifest_hash:
            raise ValueError("harmonized manifest hash mismatch")
        if (
            harmonized_summary["source_protocol_manifest_sha256"]
            != protocol_summary["manifest_sha256"]
        ):
            raise ValueError("harmonized manifest belongs to a different protocol")
    elif manifest_hash != protocol_summary["manifest_sha256"]:
        raise ValueError("unified manifest hash mismatch")
    if splits_hash != protocol_summary["splits_sha256"]:
        raise ValueError("frozen split hash mismatch")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "controlled_repair_summary.json"
    if args.resume and summary_path.is_file():
        complete = json.loads(summary_path.read_text())
        if (
            complete.get("all_gates_pass")
            and complete.get("recipe", {}).get("name") == args.recipe
            and complete.get("manifest_sha256") == manifest_hash
            and complete.get("splits_sha256") == splits_hash
        ):
            print("CONTROLLED REPAIR ALREADY COMPLETE", flush=True)
            print(json.dumps(complete, indent=2), flush=True)
            return 0

    manifest = pd.read_csv(manifest_path).sort_values("manifest_index").reset_index(drop=True)
    split_data = json.loads(splits_path.read_text())
    fold = next(row for row in split_data["folds"] if int(row["fold"]) == args.fold)
    train_subjects = set(fold["train_subjects"])
    validation_subjects = set(fold["validation_subjects"])
    test_subjects = set(fold["test_subjects"])
    if train_subjects & validation_subjects or (train_subjects | validation_subjects) & test_subjects:
        raise AssertionError("patient leakage in frozen split")
    training_eligible = manifest.training_eligible.astype(str).str.lower().isin(
        ["true", "1"]
    )
    if uses_harmonized_manifest:
        if "evaluation_eligible" not in manifest.columns:
            raise ValueError("harmonized manifest lacks evaluation_eligible")
        evaluation_eligible = manifest.evaluation_eligible.astype(str).str.lower().isin(
            ["true", "1"]
        )
    else:
        evaluation_eligible = pd.Series(True, index=manifest.index)
    train_indices = manifest.index[
        manifest.subject_id.astype(str).isin(train_subjects) & training_eligible
    ].to_numpy()
    validation_indices = manifest.index[
        manifest.subject_id.astype(str).isin(validation_subjects) & evaluation_eligible
    ].to_numpy()
    # Outer-test indices are deliberately never constructed.

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
        if checkpoint["recipe"] != args.recipe:
            raise ValueError("resume checkpoint belongs to a different recipe")
        if checkpoint["manifest_sha256"] != manifest_hash:
            raise ValueError("resume checkpoint belongs to a different manifest")
        if checkpoint["splits_sha256"] != splits_hash:
            raise ValueError("resume checkpoint belongs to different splits")
        preprocessor = HRVFoldPreprocessor.from_state_dict(checkpoint["hrv_preprocessor"])
        print(f"RESUME {args.recipe} from epoch {checkpoint['epoch'] + 2}", flush=True)
    else:
        print(f"Fit HRV preprocessing on {len(train_indices)} training windows only", flush=True)
        preprocessor = HRVFoldPreprocessor().fit(raw.raw_hrv_for_indices(train_indices))
    data = UnifiedCacheDataset(
        args.legacy_cache,
        args.iridia_cache,
        manifest_path,
        preprocessor,
        "segment_zscore",
    )
    model = build_model("proposed-groupnorm").to(device)
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
        active_bins = lead_bins_for_epoch(args.recipe, epoch)
        selection_eligible = args.recipe == "groupnorm" or epoch >= 6
        schedule = build_epoch_schedule(
            train_manifest,
            HierarchicalSamplerConfig(
                batch_size=32,
                positives_per_batch=8,
                batches_per_epoch=args.batches_per_epoch,
                positive_lead_bins=active_bins,
                stratified_negative_sources=(
                    ("iridia", "legacy") if harmonized_negatives else ("iridia",)
                ),
                seed=seed,
            ),
            epoch=epoch,
        )
        observed_bins = set(schedule.loc[schedule.label.eq(1), "lead_bin"].astype(str))
        if not observed_bins.issubset(set(active_bins)):
            raise AssertionError("sampler violated the active curriculum bins")
        batches = [
            group.sort_values("slot").manifest_index.astype(int).tolist()
            for _, group in schedule.groupby("batch_id", sort=True)
        ]
        loader = DataLoader(data, batch_sampler=batches, num_workers=0, pin_memory=True)
        model.train()
        losses, classification_losses, alignment_losses = [], [], []
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            y = batch["label"].to(device, non_blocking=True)
            ecg, hrv = apply_input_mode(
                batch["ecg"].to(device, non_blocking=True),
                batch["hrv"].to(device, non_blocking=True),
                input_mode,
            )
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits, representation = model.forward_with_representation(ecg, hrv)
                classification_loss = F.binary_cross_entropy_with_logits(logits, y)
            if use_domain_alignment:
                source_is_iridia = torch.tensor(
                    [str(source) == "iridia" for source in batch["source_family"]],
                    dtype=torch.bool,
                    device=device,
                )
                alignment_loss = conditional_domain_mean_loss(
                    representation, y.long(), source_is_iridia
                )
            else:
                alignment_loss = torch.zeros((), device=device)
            loss = classification_loss + domain_alignment_weight * alignment_loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
            classification_losses.append(float(classification_loss.detach().cpu()))
            alignment_losses.append(float(alignment_loss.detach().cpu()))
        validation, _, _, _ = evaluate(
            model, data, validation_indices, device, input_mode=input_mode
        )
        row = {
            "epoch": epoch + 1,
            "train_loss": float(np.mean(losses)),
            "classification_loss": float(np.mean(classification_losses)),
            "domain_alignment_loss": float(np.mean(alignment_losses)),
            "validation": validation,
            "active_positive_lead_bins": list(active_bins),
            "selection_eligible": selection_eligible,
            "elapsed_minutes": (time.time() - started) / 60.0,
        }
        history.append(row)
        improved = selection_eligible and validation["pr_auc"] > best_pr_auc + 1e-5
        if improved:
            best_pr_auc = validation["pr_auc"]
            stale_epochs = 0
        elif selection_eligible:
            stale_epochs += 1
        else:
            stale_epochs = 0
        payload = {
            "epoch": epoch,
            "recipe": args.recipe,
            "model_state_dict": {
                key: value.detach().cpu() for key, value in model.state_dict().items()
            },
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "hrv_preprocessor": preprocessor.state_dict(),
            "best_pr_auc": best_pr_auc,
            "stale_epochs": stale_epochs,
            "history": history,
            "manifest_sha256": manifest_hash,
            "splits_sha256": splits_hash,
            "fold": args.fold,
            "rng_state": rng_state(),
        }
        save_checkpoint(last_path, payload)
        if improved:
            save_checkpoint(best_path, payload)
        history_path.write_text(json.dumps(history, indent=2))
        print(
            f"{args.recipe} EPOCH {epoch + 1:02d}: bins={','.join(active_bins)}; "
            f"train_loss={row['train_loss']:.4f}; val_loss={validation['loss']:.4f}; "
            f"align_loss={row['domain_alignment_loss']:.4f}; "
            f"val_pr_auc={validation['pr_auc']:.4f}; val_auroc={validation['auroc']:.4f}; "
            f"selection_eligible={selection_eligible}; stale={stale_epochs}/{args.patience}",
            flush=True,
        )
        if selection_eligible and stale_epochs >= args.patience:
            print("EARLY STOP", flush=True)
            break

    if not best_path.is_file():
        raise RuntimeError("no selection-eligible best checkpoint was saved")
    best = torch.load(best_path, map_location="cpu", weights_only=False)
    model.load_state_dict(best["model_state_dict"])
    model.to(device)
    _, y_val, p_val, validation_manifest_indices = evaluate(
        model, data, validation_indices, device, input_mode=input_mode
    )
    prediction_columns = [
        "manifest_index", "window_id", "subject_id", "source_family", "label",
        "event_id", "lead_bin", "negative_stratum", "record_id",
    ]
    predictions = manifest.loc[validation_manifest_indices, prediction_columns].copy()
    predictions["probability"] = p_val
    prediction_path = args.output_dir / "best_validation_predictions.csv"
    predictions.to_csv(prediction_path, index=False)
    diagnostics = summarize_predictions(predictions)
    full_horizon_selected = (
        args.recipe == "groupnorm" or int(best["epoch"]) >= 6
    )
    if harmonized_negatives:
        legacy_negative = manifest[
            manifest.source_family.astype(str).eq("legacy")
            & manifest.label.astype(int).eq(0)
        ]
        harmonized_policy_verified = (
            set(
                legacy_negative.loc[
                    legacy_negative.training_eligible.astype(str).str.lower().isin(
                        ["true", "1"]
                    ),
                    "negative_stratum",
                ].astype(str)
            )
            == {"near_20_60", "medium_60_360", "far_360_plus_or_none"}
            and not legacy_negative.loc[
                legacy_negative.negative_stratum.astype(str).eq(
                    "post_af_recovery_excluded"
                ),
                "training_eligible",
            ].astype(str).str.lower().isin(["true", "1"]).any()
        )
    else:
        harmonized_policy_verified = True
    summary = {
        "format": "afib_controlled_validation_repair_v1",
        "fold": args.fold,
        "recipe": {
            "name": args.recipe,
            "model": "proposed-groupnorm",
            "normalization": "groupnorm",
            "input_mode": input_mode,
            "curriculum": args.recipe != "groupnorm",
            "domain_alignment": use_domain_alignment,
            "harmonized_legacy_negative_timing": uses_harmonized_manifest,
            "stratified_negative_sources": (
                ["iridia", "legacy"] if harmonized_negatives else ["iridia"]
            ),
            "domain_alignment_weight": (
                domain_alignment_weight if use_domain_alignment else 0.0
            ),
            "curriculum_stages": {
                "epochs_1_3": list(ALL_LEAD_BINS[:2]),
                "epochs_4_6": list(ALL_LEAD_BINS[:3]),
                "epochs_7_plus": list(ALL_LEAD_BINS),
            } if args.recipe != "groupnorm" else None,
            "batch_size": 32,
            "positive_per_batch": 8,
            "negative_per_batch": 24,
            "batches_per_epoch": args.batches_per_epoch,
            "learning_rate": 5e-4,
            "weight_decay": 1e-4,
            "loss": "binary_cross_entropy",
            "selection_metric": "validation_pr_auc",
            "specificity_floor": 0.90,
        },
        "device": torch.cuda.get_device_name(0),
        "parameter_count": model.parameter_count,
        "train_windows": int(len(train_indices)),
        "validation_windows": int(len(validation_indices)),
        "best_epoch": int(best["epoch"]) + 1,
        "outer_test_evaluated": False,
        "diagnostics": diagnostics,
        "manifest_sha256": manifest_hash,
        "splits_sha256": splits_hash,
        "history": history,
        "gates": {
            "outer_test_not_evaluated": True,
            "patient_disjoint": not train_subjects & validation_subjects
            and not (train_subjects | validation_subjects) & test_subjects,
            "groupnorm_installed": not any(
                isinstance(module, nn.BatchNorm1d) for module in model.modules()
            ) and any(isinstance(module, nn.GroupNorm) for module in model.modules()),
            "parameter_budget": model.parameter_count < MAX_TRAINABLE_PARAMETERS,
            "full_horizon_checkpoint_selected": full_horizon_selected,
            "finite_validation_metrics": all(
                np.isfinite(diagnostics["ranking"][key])
                for key in ("pr_auc", "auroc", "brier", "log_loss")
            ),
            "checkpoint_saved": best_path.is_file(),
            "validation_predictions_saved": prediction_path.is_file(),
            "harmonized_negative_policy_verified": bool(
                harmonized_policy_verified
            ),
        },
    }
    summary["all_gates_pass"] = all(summary["gates"].values())
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    printable = {key: value for key, value in summary.items() if key != "history"}
    print(json.dumps(printable, indent=2), flush=True)
    if not summary["all_gates_pass"]:
        raise RuntimeError("controlled repair integrity gate failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

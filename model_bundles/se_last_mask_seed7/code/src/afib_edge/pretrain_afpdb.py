"""Fixed-budget full-sequence pretraining on the labelled AFPDB PAF cohort.

PhysioNet states that its 50 learning record sets come from 48 people but does
not publish which record-set identifiers repeat. To avoid claiming an
unverifiable patient-disjoint source validation split, every labelled ``p``
pair is used only for source training. Epoch count is fixed in advance; no
source metric selects a checkpoint. Transfer is selected later using the
patient-grouped AFDB/LTAFDB development folds.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from scipy.special import expit
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader

from .afpdb import CACHE_FORMAT
from .dataset import CacheWindowDataset
from .hrv_preprocess import HRVFoldPreprocessor
from .icentia import atomic_json, digest_file
from .models import ModelConfig, build_model
from .training import resolve_device


CHECKPOINT_FORMAT = "afpdb_se_sequence_pretraining_v1"


def atomic_torch(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def cpu_state(module: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in module.state_dict().items()}


def representation_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
        if name.startswith(("ecg_encoder.", "temporal."))
    }


def validate_cache(cache_root: Path) -> dict:
    summary_path = cache_root / "summary.json"
    manifest_path = cache_root / "windows.csv"
    if not (cache_root / "COMPLETE").is_file() or not summary_path.is_file():
        raise ValueError("AFPDB cache is incomplete")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("format") != CACHE_FORMAT:
        raise ValueError("unrecognized AFPDB cache format")
    if digest_file(manifest_path) != summary.get("manifest_sha256"):
        raise ValueError("AFPDB manifest checksum mismatch")
    if int(summary["record_pair_groups"]) != 25 or summary["class_counts"] != [50, 50]:
        raise ValueError("AFPDB cache lacks the expected 25 balanced PAF record pairs")
    return summary


def mask_contiguous_ecg(
    ecg: torch.Tensor,
    generator: torch.Generator,
    probability: float = 0.5,
) -> torch.Tensor:
    masked = ecg.clone()
    selected = torch.rand(ecg.shape[0], generator=generator) < probability
    for batch_index in selected.nonzero(as_tuple=False).flatten().tolist():
        segment = int(torch.randint(0, 20, (), generator=generator).item())
        length = int(torch.randint(125, 501, (), generator=generator).item())
        start = int(torch.randint(0, 7500 - length + 1, (), generator=generator).item())
        masked[batch_index, segment, 0, start : start + length] = 0.0
    return masked


def evaluate_training_set(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, float, float]:
    """Return descriptive in-sample metrics; never use them for checkpoint selection."""
    model.eval()
    labels, logits = [], []
    total_loss, samples = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            ecg = batch["ecg"].to(device, non_blocking=device.type == "cuda")
            hrv = batch["hrv"].to(device, non_blocking=device.type == "cuda")
            target = batch["label"].to(device, non_blocking=device.type == "cuda")
            output = model(ecg, hrv)
            loss = nn.functional.binary_cross_entropy_with_logits(output, target)
            total_loss += float(loss) * len(target)
            samples += len(target)
            labels.append(target.cpu().numpy())
            logits.append(output.cpu().numpy())
    y = np.concatenate(labels)
    probability = expit(np.concatenate(logits))
    return (
        total_loss / samples,
        float(average_precision_score(y, probability)),
        float(roc_auc_score(y, probability)),
    )


def run(args: argparse.Namespace) -> dict:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = validate_cache(args.cache_root)
    manifest = args.cache_root / "windows.csv"
    raw = CacheWindowDataset(args.cache_root, manifest, ecg_normalization="none")
    all_indices = np.arange(len(raw), dtype=np.int64)
    processor = HRVFoldPreprocessor().fit(raw.raw_hrv_for_indices(all_indices))
    data = CacheWindowDataset(
        args.cache_root,
        manifest,
        hrv_preprocessor=processor,
        ecg_normalization="segment_zscore",
    )
    if set(np.unique(data.labels)) != {0, 1}:
        raise ValueError("AFPDB source training requires both classes")

    config = {
        "cache_summary_sha256": digest_file(args.cache_root / "summary.json"),
        "manifest_sha256": digest_file(manifest),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "device": str(resolve_device(args.device)),
        "max_train_batches": args.max_train_batches,
        "model_config": asdict(ModelConfig()),
        "ecg_normalization": "segment_zscore",
        "hrv_preprocessing": "all source-training records median/mean/std",
        "training_augmentation": "one random 0.5-2.0 second mask with probability 0.5",
        "checkpoint_selection": "fixed final epoch; no source validation split",
        "source_identity_caveat": (
            "PhysioNet documents 50 learning record sets from 48 people but does not "
            "publish duplicate-pair identities; therefore no source validation claim"
        ),
        "torch_version": torch.__version__,
    }
    signature = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    experiment = {
        "format": CHECKPOINT_FORMAT,
        "signature": signature,
        "config": config,
        "cache_summary": summary,
        "source_record_pair_groups": sorted(map(str, np.unique(raw.subject_ids))),
        "source_samples": len(data),
        "hrv_preprocessor": processor.state_dict(),
    }
    experiment_path = args.output_dir / "experiment.json"
    if experiment_path.is_file():
        if json.loads(experiment_path.read_text(encoding="utf-8")) != experiment:
            raise ValueError("AFPDB pretraining plan changed; use a new output directory")
    atomic_json(experiment_path, experiment)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = resolve_device(args.device)
    loader_generator = torch.Generator().manual_seed(args.seed)
    augmentation_generator = torch.Generator().manual_seed(args.seed + 1009)
    train_loader = DataLoader(
        data,
        batch_size=args.batch_size,
        shuffle=True,
        generator=loader_generator,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    evaluation_loader = DataLoader(
        data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    model = build_model("proposed").to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    latest = args.output_dir / "latest.pt"
    history: list[dict] = []
    start_epoch = 1
    if args.resume and latest.is_file():
        saved = torch.load(latest, map_location="cpu", weights_only=True)
        if saved["signature"] != signature:
            raise ValueError("AFPDB resume mismatch; use a new output directory")
        model.load_state_dict(saved["model_state_dict"])
        optimizer.load_state_dict(saved["optimizer_state_dict"])
        for state in optimizer.state.values():
            for name, value in state.items():
                if torch.is_tensor(value):
                    state[name] = value.to(device)
        history = saved["history"]
        start_epoch = int(saved["epoch"]) + 1
        loader_generator.set_state(saved["loader_rng"])
        augmentation_generator.set_state(saved["augmentation_rng"])
        torch.set_rng_state(saved["torch_rng"])
        if device.type == "cuda" and saved["cuda_rng"]:
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
        print(f"RESUME AFPDB from completed epoch {saved['epoch']}", flush=True)

    started = time.monotonic()
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        epoch_started = time.monotonic()
        running, seen = 0.0, 0
        for batch_index, batch in enumerate(train_loader):
            if args.max_train_batches and batch_index >= args.max_train_batches:
                break
            ecg = batch["ecg"].to(device, non_blocking=device.type == "cuda")
            hrv = batch["hrv"].to(device, non_blocking=device.type == "cuda")
            target = batch["label"].to(device, non_blocking=device.type == "cuda")
            ecg = mask_contiguous_ecg(ecg, augmentation_generator)
            optimizer.zero_grad(set_to_none=True)
            logits = model(ecg, hrv)
            loss = nn.functional.binary_cross_entropy_with_logits(logits, target)
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite AFPDB pretraining loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            running += float(loss.detach()) * len(target)
            seen += len(target)
            if batch_index == 0 or batch_index + 1 == len(train_loader):
                elapsed = time.monotonic() - started
                print(
                    f"AFPDB TRAIN epoch={epoch}/{args.epochs} "
                    f"batch={batch_index + 1}/{len(train_loader)} "
                    f"samples={seen}; elapsed={elapsed / 60:.1f} min",
                    flush=True,
                )
                atomic_json(
                    args.output_dir / "status.json",
                    {
                        "state": "running",
                        "pid": os.getpid(),
                        "phase": "fixed_budget_training",
                        "epoch": epoch,
                        "batch": batch_index + 1,
                        "batches": len(train_loader),
                        "elapsed_seconds": elapsed,
                        "updated_utc": datetime.now(UTC).isoformat(),
                    },
                )
        if seen == 0:
            raise ValueError("empty AFPDB training loader")
        in_sample_loss, in_sample_ap, in_sample_auroc = evaluate_training_set(
            model, evaluation_loader, device
        )
        row = {
            "epoch": epoch,
            "augmented_train_loss": running / seen,
            "descriptive_in_sample_loss": in_sample_loss,
            "descriptive_in_sample_ap": in_sample_ap,
            "descriptive_in_sample_auroc": in_sample_auroc,
            "epoch_seconds": time.monotonic() - epoch_started,
        }
        history.append(row)
        atomic_torch(
            latest,
            {
                "signature": signature,
                "epoch": epoch,
                "history": history,
                "model_state_dict": cpu_state(model),
                "optimizer_state_dict": optimizer.state_dict(),
                "loader_rng": loader_generator.get_state(),
                "augmentation_rng": augmentation_generator.get_state(),
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
            },
        )
        print(
            f"AFPDB EPOCH {epoch:03d}: augmented_loss={row['augmented_train_loss']:.4f} "
            f"in_sample_AP={in_sample_ap:.4f} in_sample_AUROC={in_sample_auroc:.4f}; "
            "checkpoint_policy=fixed_final_epoch",
            flush=True,
        )

    if not history:
        raise RuntimeError("AFPDB pretraining produced no completed epoch")
    checkpoint_path = args.output_dir / "sequence_final.pt"
    state = cpu_state(model)
    checkpoint = {
        "format": CHECKPOINT_FORMAT,
        "signature": signature,
        "model_config": asdict(model.config),
        "model_state_dict": state,
        "representation_state_dict": representation_state(model),
        "fixed_final_epoch": int(history[-1]["epoch"]),
        "history": history,
        "source_record_pair_groups": experiment["source_record_pair_groups"],
        "source_hrv_preprocessor": processor.state_dict(),
        "source_metrics_are_in_sample_only": True,
    }
    atomic_torch(checkpoint_path, checkpoint)
    result = {
        "format": CHECKPOINT_FORMAT,
        "all_jobs_complete": int(history[-1]["epoch"]) == args.epochs,
        "smoke_test": args.max_train_batches is not None,
        "fixed_final_epoch": int(history[-1]["epoch"]),
        "source_samples": len(data),
        "source_record_pairs": len(experiment["source_record_pair_groups"]),
        "source_metrics_are_in_sample_only": True,
        "elapsed_seconds": time.monotonic() - started,
        "checkpoint": checkpoint_path.name,
        "checkpoint_sha256": digest_file(checkpoint_path),
        "history": history,
    }
    atomic_json(args.output_dir / "summary.json", result)
    atomic_json(args.output_dir / "status.json", {"state": "finished", **result})
    print(json.dumps({key: value for key, value in result.items() if key != "history"}, indent=2))
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cuda")
    parser.add_argument("--max-train-batches", type=int, help="SMOKE TEST ONLY")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if min(args.epochs, args.batch_size) < 1:
        parser.error("epochs and batch size must be positive")
    if args.max_train_batches is not None and args.max_train_batches < 1:
        parser.error("max-train-batches must be positive")
    run(args)


if __name__ == "__main__":
    main()

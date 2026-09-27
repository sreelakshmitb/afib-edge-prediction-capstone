"""Patient-disjoint Icentia11k pretraining for the SE ECG segment encoder."""

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
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .icentia import CACHE_FORMAT, digest_file
from .models import ECGSegmentEncoder, ModelConfig
from .training import resolve_device


CHECKPOINT_FORMAT = "icentia_se_encoder_pretraining_v1"


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def atomic_torch(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def cpu_state(module: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in module.state_dict().items()}


class SegmentDataset(Dataset):
    def __init__(self, cache_root: Path, indices: np.ndarray) -> None:
        self.ecg = np.load(cache_root / "ecg.npy", mmap_mode="r")
        self.labels = np.load(cache_root / "labels.npy", mmap_mode="r")
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        index = int(self.indices[item])
        signal = np.asarray(self.ecg[index], dtype=np.float32).copy()
        mean = float(signal.mean())
        std = float(signal.std())
        signal = (signal - mean) / max(std, 1e-6)
        return {
            "ecg": torch.from_numpy(signal[None, :]),
            "label": torch.tensor(float(self.labels[index]), dtype=torch.float32),
        }


class EncoderClassifier(nn.Module):
    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or ModelConfig()
        self.encoder = ECGSegmentEncoder(self.config, use_se=True)
        self.dropout = nn.Dropout(self.config.dropout)
        self.classifier = nn.Linear(self.config.embedding_dim, 1)

    def forward(self, ecg: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.dropout(self.encoder(ecg))).squeeze(-1)


def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray, float]:
    model.eval()
    labels, probabilities = [], []
    total_loss, count = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            ecg = batch["ecg"].to(device, non_blocking=device.type == "cuda")
            target = batch["label"].to(device, non_blocking=device.type == "cuda")
            logits = model(ecg)
            loss = nn.functional.binary_cross_entropy_with_logits(logits, target)
            total_loss += float(loss) * len(target)
            count += len(target)
            labels.append(target.cpu().numpy())
            probabilities.append(torch.sigmoid(logits).cpu().numpy())
    if count == 0:
        raise ValueError("empty validation loader")
    return np.concatenate(labels), np.concatenate(probabilities), total_loss / count


def validate_cache(cache_root: Path) -> dict:
    complete = cache_root / "COMPLETE"
    metadata_path = cache_root / "metadata.json"
    if not complete.is_file() or not metadata_path.is_file():
        raise ValueError("Icentia cache is incomplete")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("format") != CACHE_FORMAT:
        raise ValueError("unrecognized Icentia cache format")
    for filename, checksum in metadata["sha256"].items():
        path = cache_root / filename
        if not path.is_file() or digest_file(path) != checksum:
            raise ValueError(f"Icentia cache checksum mismatch: {filename}")
    return metadata


def split_indices(cache_root: Path, seed: int, validation_fraction: float) -> tuple[np.ndarray, np.ndarray]:
    labels = np.load(cache_root / "labels.npy")
    groups = np.load(cache_root / "patient_ids.npy")
    indices = np.arange(len(labels), dtype=np.int64)
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=validation_fraction,
        random_state=seed,
    )
    train, validation = next(splitter.split(indices, labels, groups))
    train = indices[train]
    validation = indices[validation]
    if set(groups[train]) & set(groups[validation]):
        raise AssertionError("Icentia patient leakage")
    for name, part in (("train", train), ("validation", validation)):
        if len(np.unique(labels[part])) != 2:
            raise ValueError(f"Icentia {name} split lacks a class")
    return train, validation


def run(args: argparse.Namespace) -> dict:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = validate_cache(args.cache_root)
    train_indices, validation_indices = split_indices(
        args.cache_root, args.seed, args.validation_fraction
    )
    labels = np.load(args.cache_root / "labels.npy")
    patients = np.load(args.cache_root / "patient_ids.npy")
    config = {
        "cache_metadata_sha256": digest_file(args.cache_root / "metadata.json"),
        "epochs": args.epochs,
        "patience": args.patience,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "validation_fraction": args.validation_fraction,
        "device": str(resolve_device(args.device)),
        "max_train_batches": args.max_train_batches,
        "model_config": asdict(ModelConfig()),
        "torch_version": torch.__version__,
    }
    signature = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    experiment = {
        "format": CHECKPOINT_FORMAT,
        "signature": signature,
        "config": config,
        "cache_summary": metadata,
        "train_indices": train_indices.tolist(),
        "validation_indices": validation_indices.tolist(),
        "train_patients": sorted(map(str, np.unique(patients[train_indices]))),
        "validation_patients": sorted(map(str, np.unique(patients[validation_indices]))),
    }
    experiment_path = args.output_dir / "experiment.json"
    if experiment_path.is_file():
        previous = json.loads(experiment_path.read_text(encoding="utf-8"))
        if previous != experiment:
            raise ValueError("pretraining plan changed; use a new output directory")
    atomic_json(experiment_path, experiment)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = resolve_device(args.device)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        SegmentDataset(args.cache_root, train_indices),
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    validation_loader = DataLoader(
        SegmentDataset(args.cache_root, validation_indices),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    model = EncoderClassifier().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    class_counts = np.bincount(labels[train_indices], minlength=2)
    positive_weight = torch.tensor(
        float(class_counts[0] / max(class_counts[1], 1)), device=device
    )
    latest = args.output_dir / "latest.pt"
    history: list[dict] = []
    best_state = None
    best_epoch, best_auroc, stale, start_epoch = 0, -1.0, 0, 1
    if args.resume and latest.is_file():
        saved = torch.load(latest, map_location="cpu", weights_only=True)
        if saved["signature"] != signature:
            raise ValueError("resume mismatch; use a new output directory")
        model.load_state_dict(saved["model_state_dict"])
        optimizer.load_state_dict(saved["optimizer_state_dict"])
        for state in optimizer.state.values():
            for name, value in state.items():
                if torch.is_tensor(value):
                    state[name] = value.to(device)
        history = saved["history"]
        best_state = saved["best_state_dict"]
        best_epoch = int(saved["best_epoch"])
        best_auroc = float(saved["best_auroc"])
        stale = int(saved["stale"])
        start_epoch = int(saved["epoch"]) + 1
        generator.set_state(saved["loader_rng"])
        torch.set_rng_state(saved["torch_rng"])
        if device.type == "cuda" and saved["cuda_rng"]:
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
        print(f"RESUME from completed epoch {saved['epoch']}", flush=True)

    started = time.monotonic()
    for epoch in range(start_epoch, args.epochs + 1):
        if stale >= args.patience:
            break
        model.train()
        epoch_start = time.monotonic()
        running, seen = 0.0, 0
        for batch_index, batch in enumerate(train_loader):
            if args.max_train_batches and batch_index >= args.max_train_batches:
                break
            ecg = batch["ecg"].to(device, non_blocking=device.type == "cuda")
            target = batch["label"].to(device, non_blocking=device.type == "cuda")
            optimizer.zero_grad(set_to_none=True)
            logits = model(ecg)
            loss = nn.functional.binary_cross_entropy_with_logits(
                logits, target, pos_weight=positive_weight
            )
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite pretraining loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            running += float(loss.detach()) * len(target)
            seen += len(target)
            if batch_index == 0 or batch_index + 1 == len(train_loader) or (batch_index + 1) % 25 == 0:
                elapsed = time.monotonic() - started
                print(
                    f"TRAIN epoch={epoch}/{args.epochs} batch={batch_index + 1}/"
                    f"{len(train_loader)} samples={seen} elapsed={elapsed / 60:.1f} min",
                    flush=True,
                )
                atomic_json(
                    args.output_dir / "status.json",
                    {
                        "state": "running",
                        "pid": os.getpid(),
                        "phase": "training",
                        "epoch": epoch,
                        "batch": batch_index + 1,
                        "batches": len(train_loader),
                        "elapsed_seconds": elapsed,
                        "updated_utc": datetime.now(UTC).isoformat(),
                    },
                )
        if seen == 0:
            raise ValueError("empty Icentia training loader")
        y, probability, validation_loss = predict(model, validation_loader, device)
        auroc = float(roc_auc_score(y, probability))
        ap = float(average_precision_score(y, probability))
        if auroc > best_auroc:
            best_auroc, best_epoch, stale = auroc, epoch, 0
            best_state = cpu_state(model)
        else:
            stale += 1
        row = {
            "epoch": epoch,
            "train_loss": running / seen,
            "validation_loss": validation_loss,
            "validation_auroc": auroc,
            "validation_ap": ap,
            "epoch_seconds": time.monotonic() - epoch_start,
        }
        history.append(row)
        atomic_torch(
            latest,
            {
                "signature": signature,
                "epoch": epoch,
                "history": history,
                "model_state_dict": cpu_state(model),
                "best_state_dict": best_state,
                "optimizer_state_dict": optimizer.state_dict(),
                "best_epoch": best_epoch,
                "best_auroc": best_auroc,
                "stale": stale,
                "loader_rng": generator.get_state(),
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
            },
        )
        print(
            f"EPOCH {epoch:03d}: train_loss={row['train_loss']:.4f} "
            f"val_loss={validation_loss:.4f} val_AUROC={auroc:.4f} val_AP={ap:.4f}; "
            f"best_epoch={best_epoch}; stale={stale}/{args.patience}; "
            f"epoch_time={row['epoch_seconds']:.1f}s",
            flush=True,
        )

    if best_state is None:
        raise RuntimeError("pretraining produced no valid checkpoint")
    model.load_state_dict(best_state)
    checkpoint_path = args.output_dir / "encoder_best.pt"
    checkpoint = {
        "format": CHECKPOINT_FORMAT,
        "signature": signature,
        "model_config": asdict(model.config),
        "encoder_state_dict": cpu_state(model.encoder),
        "best_epoch": best_epoch,
        "best_validation_auroc": best_auroc,
        "history": history,
        "train_patients": experiment["train_patients"],
        "validation_patients": experiment["validation_patients"],
        "source_cache_metadata_sha256": config["cache_metadata_sha256"],
    }
    atomic_torch(checkpoint_path, checkpoint)
    summary = {
        "format": CHECKPOINT_FORMAT,
        "all_jobs_complete": True,
        "smoke_test": args.max_train_batches is not None,
        "best_epoch": best_epoch,
        "best_validation_auroc": best_auroc,
        "train_samples": int(len(train_indices)),
        "validation_samples": int(len(validation_indices)),
        "train_patients": len(experiment["train_patients"]),
        "validation_patients": len(experiment["validation_patients"]),
        "elapsed_seconds": time.monotonic() - started,
        "checkpoint": checkpoint_path.name,
        "checkpoint_sha256": digest_file(checkpoint_path),
        "history": history,
    }
    atomic_json(args.output_dir / "summary.json", summary)
    atomic_json(
        args.output_dir / "status.json",
        {"state": "finished", "phase": "complete", **summary},
    )
    print(json.dumps({key: value for key, value in summary.items() if key != "history"}, indent=2))
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cuda")
    parser.add_argument("--max-train-batches", type=int, help="SMOKE TEST ONLY")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if min(args.epochs, args.patience, args.batch_size) < 1:
        parser.error("epochs, patience, and batch size must be positive")
    if not 0.0 < args.validation_fraction < 0.5:
        parser.error("validation fraction must be in (0, 0.5)")
    if args.max_train_batches is not None and args.max_train_batches < 1:
        parser.error("max train batches must be positive")
    run(args)


if __name__ == "__main__":
    main()

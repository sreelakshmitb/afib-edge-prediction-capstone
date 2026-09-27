"""Resumable prediction quality sprint using inner validation only.

All candidates use the same five inner GroupKFold splits of outer fold 0's
development partition. The V9 comparison keeps architecture, natural window
weighting, augmentation, folds, seed, and maximum epoch budget fixed while
isolating a pre-specified AP-aligned term on the winning masked-BCE baseline.
Validation always uses unweighted 20-minute labels at natural prevalence. There
is deliberately no outer-test inference option.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import gc
import hashlib
import json
import os
import random
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import sklearn
import torch
from scipy.special import expit
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .curriculum import build_horizon_labels, curriculum_schedule
from .dataset import CacheWindowDataset
from .hrv_preprocess import HRVFoldPreprocessor
from .patient_objectives import (
    SMOOTH_AP_COEFFICIENT,
    SMOOTH_AP_TEMPERATURE,
    patient_equal_weights,
    smooth_binary_average_precision_loss,
)
from .sprint_models import (
    build_sprint_model,
    uses_ap_aligned_loss,
    uses_afpdb_pretraining,
    uses_ecg_masking,
    uses_focal_loss,
    uses_horizon_curriculum,
    uses_icentia_pretraining,
    uses_patient_equal_loss,
)
from .training import (
    BinaryFocalLoss,
    classification_metrics,
    make_outer_fold_indices,
    make_validation_split,
    ranking_metrics,
    resolve_device,
    select_f1_threshold,
)

CANDIDATES = (
    "paper_control",
    "se_last_natural",
    "se_last_mask",
    "se_last_mask_ap",
    "se_last_mask_curriculum",
    "se_last_mask_patient_equal",
    "se_last_mask_patient_equal_ap",
    "se_last_mask_afpdb",
    "se_last_mask_icentia",
    "se_last_focal",
    "se_last_focal_mask",
)
PARENT_CANDIDATE = {
    "paper_control": None,
    "se_last_natural": "paper_control",
    "se_last_mask": "se_last_natural",
    "se_last_mask_ap": "se_last_mask",
    "se_last_mask_curriculum": "se_last_mask",
    "se_last_mask_patient_equal": "se_last_mask",
    "se_last_mask_patient_equal_ap": "se_last_mask",
    "se_last_mask_afpdb": "se_last_mask",
    "se_last_mask_icentia": "se_last_mask",
    "se_last_focal": "se_last_natural",
    "se_last_focal_mask": "se_last_focal",
}
VERSION = "prediction_quality_sprint_v9_ap_only"


def digest_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def atomic_torch(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def cpu_state(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


class Progress:
    """Persist progress and report process liveness separately from advancement."""

    def __init__(self, directory, heartbeat_seconds=30):
        self.directory = Path(directory)
        self.started = time.monotonic()
        self.state = {"state": "running", "pid": os.getpid(), "phase": "startup"}
        self.lock = threading.RLock()
        self.stop = threading.Event()
        self.interval = heartbeat_seconds
        self.thread = threading.Thread(target=self._heartbeat, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def emit(self, message):
        with self.lock:
            stamp = datetime.now(UTC).isoformat(timespec="seconds")
            line = f"[{stamp}] {message}"
            print(line, flush=True)
            with (self.directory / "progress.log").open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")

    def update(self, **fields):
        with self.lock:
            self.state.update(fields)
            self.state["last_progress_utc"] = datetime.now(UTC).isoformat()

    def _heartbeat(self):
        while not self.stop.wait(self.interval):
            with self.lock:
                self.state["heartbeat_utc"] = datetime.now(UTC).isoformat()
                self.state["elapsed_seconds"] = time.monotonic() - self.started
                atomic_json(self.directory / "status.json", self.state)
                self.emit(
                    f"HEARTBEAT process alive; phase={self.state.get('phase')}; "
                    f"job={self.state.get('job', 'setup')}; "
                    f"epoch={self.state.get('epoch', 0)}; "
                    f"batch={self.state.get('batch', 0)}/{self.state.get('batches', 0)}; "
                    f"elapsed={self.state['elapsed_seconds'] / 60:.1f} min"
                )

    def __exit__(self, exc_type, exc, traceback):
        self.stop.set()
        self.thread.join(timeout=2)
        with self.lock:
            self.state.update(
                state="finished" if exc_type is None else "interrupted_or_failed",
                error=str(exc) if exc_type else None,
                heartbeat_utc=datetime.now(UTC).isoformat(),
                elapsed_seconds=time.monotonic() - self.started,
            )
            atomic_json(self.directory / "status.json", self.state)


class DevelopmentData:
    """Reject signal/HRV requests for every excluded outer-test index."""

    def __init__(self, base, indices, horizon_labels=None):
        self.base = base
        self.labels = base.labels
        self.subject_ids = base.subject_ids
        self.refs = base.window_refs
        self.indices = np.asarray(indices, dtype=np.int64)
        self.positions = {int(index): pos for pos, index in enumerate(self.indices)}
        self.raw_hrv = base.raw_hrv_for_indices(self.indices)
        self.accessed = set()
        self.horizon_labels = horizon_labels

    def check(self, indices):
        if not set(map(int, indices)).issubset(self.positions):
            raise AssertionError("outer-test access is forbidden in the improvement sprint")

    def hrv(self, indices):
        self.check(indices)
        return self.raw_hrv[[self.positions[int(index)] for index in indices]]

    def sample(self, index):
        self.check([index])
        self.accessed.add(int(index))
        return self.base[int(index)]


class FoldView(Dataset):
    def __init__(self, data, indices, hrv, candidate, labels=None, sample_weights=None):
        data.check(indices)
        self.data, self.indices, self.hrv, self.candidate = data, indices, hrv, candidate
        self.labels = data.labels if labels is None else np.asarray(labels, dtype=np.uint8)
        if len(self.labels) != len(data.labels):
            raise ValueError("fold-view labels must align with the complete dataset")
        self.sample_weights = (
            np.ones(len(data.labels), dtype=np.float32)
            if sample_weights is None
            else np.asarray(sample_weights, dtype=np.float32)
        )
        if len(self.sample_weights) != len(data.labels):
            raise ValueError("fold-view weights must align with the complete dataset")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        index = int(self.indices[item])
        self.data.check([index])
        ecg = self.data.sample(index)["ecg"]
        return {
            "ecg": ecg,
            "hrv": torch.from_numpy(self.hrv[item]),
            "label": torch.tensor(float(self.labels[index]), dtype=torch.float32),
            "sample_weight": torch.tensor(
                float(self.sample_weights[index]), dtype=torch.float32
            ),
        }


def preprocess(data, train_indices, validation_indices, candidate):
    if candidate not in CANDIDATES:
        raise ValueError(f"unknown candidate: {candidate}")
    # Earlier screens found no meaningful gain from winsor/log HRV processing.
    # Hold preprocessing fixed so the current comparison isolates initialization.
    processor = HRVFoldPreprocessor()
    processor.fit(data.hrv(train_indices))
    state = processor.state_dict()
    state.setdefault("kind", "original_standardize")
    return (
        processor.transform(data.hrv(train_indices)),
        processor.transform(data.hrv(validation_indices)),
        state,
    )


def mask_contiguous_ecg(
    ecg,
    probability=0.5,
    minimum_samples=125,
    maximum_samples=500,
    generator=None,
):
    """Mask one 0.5-2.0 s interval in a randomly selected segment.

    The ECG has already been segment-wise z-scored, so zero is the segment
    mean.  The operation is training-only and preserves the input shape.
    """
    if ecg.ndim != 4 or tuple(ecg.shape[1:]) != (20, 1, 7500):
        raise ValueError("ECG must have shape [B,20,1,7500]")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("mask probability must be in [0,1]")
    if not 1 <= minimum_samples <= maximum_samples <= ecg.shape[-1]:
        raise ValueError("invalid contiguous-mask length")
    masked = ecg.clone()
    # Draw indices on CPU even for CUDA training. This keeps augmentation on a
    # dedicated reproducible RNG stream and avoids a GPU synchronization for
    # every scalar index sampled below.
    selected = torch.rand(ecg.shape[0], generator=generator) < probability
    for batch_index in selected.nonzero(as_tuple=False).flatten().tolist():
        segment = int(
            torch.randint(
                0,
                ecg.shape[1],
                (),
                generator=generator,
            ).item()
        )
        length = int(
            torch.randint(
                minimum_samples,
                maximum_samples + 1,
                (),
                generator=generator,
            ).item()
        )
        start = int(
            torch.randint(
                0,
                ecg.shape[-1] - length + 1,
                (),
                generator=generator,
            ).item()
        )
        masked[batch_index, segment, 0, start : start + length] = 0.0
    return masked


def candidate_recipe(candidate):
    return {
        "architecture": "paper" if candidate == "paper_control" else "se",
        "hrv_preprocessing": "training_median_mean_std",
        "sampling": "natural",
        "temporal_pooling": "last",
        "loss": (
            (
                "patient_weighted_bce_plus_smooth_ap"
                if uses_patient_equal_loss(candidate)
                else "bce_plus_smooth_ap"
            )
            if uses_ap_aligned_loss(candidate)
            else (
                "patient_weighted_bce"
                if uses_patient_equal_loss(candidate)
                else ("focal" if uses_focal_loss(candidate) else "bce")
            )
        ),
        "focal_gamma": 2.0 if uses_focal_loss(candidate) else None,
        "focal_alpha": None,
        "ecg_masking": bool(uses_ecg_masking(candidate)),
        "ecg_mask_probability": 0.5 if uses_ecg_masking(candidate) else 0.0,
        "ecg_mask_samples": [125, 500] if uses_ecg_masking(candidate) else None,
        "horizon_curriculum": (
            {
                "training_horizons_minutes": [5, 10, 15, 20],
                "warmup_epochs_per_shorter_horizon": 2,
                "validation_horizon_minutes": 20,
                "best_checkpoint_from_20_minute_stage_only": True,
            }
            if uses_horizon_curriculum(candidate)
            else None
        ),
        "patient_equal_loss": bool(uses_patient_equal_loss(candidate)),
        "patient_weight_formula": (
            "N_train/(N_patients*windows_for_patient)"
            if uses_patient_equal_loss(candidate)
            else None
        ),
        "ap_aligned_objective": (
            {
                "kind": "within_batch_smooth_binary_average_precision",
                "coefficient": SMOOTH_AP_COEFFICIENT,
                "temperature": SMOOTH_AP_TEMPERATURE,
                "fallback_without_both_classes": (
                    "patient_weighted_bce_only"
                    if uses_patient_equal_loss(candidate)
                    else "bce_only"
                ),
            }
            if uses_ap_aligned_loss(candidate)
            else None
        ),
        "encoder_initialization": (
            "patient-disjoint_AFPDB_sequence_pretraining"
            if uses_afpdb_pretraining(candidate)
            else (
                "patient-disjoint_Icentia11k_rhythm_pretraining"
                if uses_icentia_pretraining(candidate)
                else "random"
            )
        ),
        "transferred_components": (
            ["ecg_encoder", "unidirectional_lstm"]
            if uses_afpdb_pretraining(candidate)
            else (["ecg_encoder"] if uses_icentia_pretraining(candidate) else [])
        ),
        "batchnorm_running_statistics": (
            "reset_after_transfer" if uses_afpdb_pretraining(candidate) else "default"
        ),
    }


def load_pretrained_encoder(model, checkpoint_path):
    """Load only the ECG encoder and reject incompatible checkpoints."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("format") != "icentia_se_encoder_pretraining_v1":
        raise ValueError("unrecognized Icentia encoder checkpoint")
    state = checkpoint.get("encoder_state_dict")
    if not isinstance(state, dict) or not state:
        raise ValueError("Icentia checkpoint has no encoder state")
    expected = model.ecg_encoder.state_dict()
    if set(state) != set(expected):
        missing = sorted(set(expected) - set(state))
        extra = sorted(set(state) - set(expected))
        raise ValueError(f"encoder keys mismatch; missing={missing}, extra={extra}")
    for name, value in state.items():
        if tuple(value.shape) != tuple(expected[name].shape):
            raise ValueError(f"encoder tensor shape mismatch for {name}")
    model.ecg_encoder.load_state_dict(state, strict=True)
    return checkpoint


def load_afpdb_representation(model, checkpoint_path):
    """Load the ECG encoder and UniLSTM, then reset source-domain BN statistics."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("format") != "afpdb_se_sequence_pretraining_v1":
        raise ValueError("unrecognized AFPDB sequence checkpoint")
    state = checkpoint.get("representation_state_dict")
    if not isinstance(state, dict) or not state:
        raise ValueError("AFPDB checkpoint has no representation state")
    expected = {
        name: value
        for name, value in model.state_dict().items()
        if name.startswith(("ecg_encoder.", "temporal."))
    }
    if set(state) != set(expected):
        missing = sorted(set(expected) - set(state))
        extra = sorted(set(state) - set(expected))
        raise ValueError(f"AFPDB representation keys mismatch; missing={missing}, extra={extra}")
    for name, value in state.items():
        if tuple(value.shape) != tuple(expected[name].shape):
            raise ValueError(f"AFPDB representation tensor shape mismatch for {name}")
    current = model.state_dict()
    current.update(state)
    model.load_state_dict(current, strict=True)
    for module in model.ecg_encoder.modules():
        if isinstance(module, nn.BatchNorm1d):
            module.reset_running_stats()
    return checkpoint


def run_epoch(
    model,
    loader,
    device,
    progress,
    optimizer=None,
    max_batches=None,
    mask_ecg=False,
    augmentation_generator=None,
    training_criterion=None,
    patient_equal_loss=False,
    ap_loss_coefficient=0.0,
    ap_loss_temperature=SMOOTH_AP_TEMPERATURE,
):
    training = optimizer is not None
    model.train(training)
    labels, logits_all = [], []
    total_loss = 0.0
    total_bce = 0.0
    total_weighted_bce = 0.0
    total_ap_loss = 0.0
    samples = 0
    last_print = time.monotonic()
    with torch.set_grad_enabled(training):
        for batch_index, batch in enumerate(loader):
            if training and max_batches and batch_index >= max_batches:
                break
            non_blocking = device.type == "cuda"
            ecg = batch["ecg"].to(device, non_blocking=non_blocking)
            hrv = batch["hrv"].to(device, non_blocking=non_blocking)
            y = batch["label"].to(device, non_blocking=non_blocking)
            sample_weight = batch["sample_weight"].to(
                device, non_blocking=non_blocking
            )
            if training and mask_ecg:
                ecg = mask_contiguous_ecg(ecg, generator=augmentation_generator)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(ecg, hrv)
            bce_per_window = nn.functional.binary_cross_entropy_with_logits(
                logits, y, reduction="none"
            )
            bce = bce_per_window.mean()
            if training and patient_equal_loss:
                if not torch.isfinite(sample_weight).all() or bool(
                    torch.any(sample_weight <= 0)
                ):
                    raise FloatingPointError("training weights must be positive and finite")
                weighted_bce = (bce_per_window * sample_weight).mean()
            else:
                weighted_bce = bce
            loss = (
                training_criterion(logits, y)
                if training and training_criterion is not None
                else weighted_bce
            )
            ap_loss = logits.sum() * 0.0
            if training and ap_loss_coefficient > 0:
                ap_loss = smooth_binary_average_precision_loss(
                    logits,
                    y,
                    temperature=ap_loss_temperature,
                    sample_weights=sample_weight,
                )
                loss = loss + ap_loss_coefficient * ap_loss
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite loss; stop and inspect data")
            if training:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                optimizer.step()
            total_loss += float(loss.detach()) * len(y)
            total_bce += float(bce.detach()) * len(y)
            total_weighted_bce += float(weighted_bce.detach()) * len(y)
            total_ap_loss += float(ap_loss.detach()) * len(y)
            samples += len(y)
            if not training:
                labels.append(y.detach().cpu().numpy())
                logits_all.append(logits.detach().cpu().numpy())
            progress.update(
                phase="training" if training else "validation",
                batch=batch_index + 1,
                batches=len(loader),
            )
            now = time.monotonic()
            if batch_index == 0 or batch_index + 1 == len(loader) or now - last_print >= 15:
                progress.emit(
                    f"{'TRAIN' if training else 'VALIDATE'} batch "
                    f"{batch_index + 1}/{len(loader)}; windows={samples}"
                )
                last_print = now
    if samples == 0:
        raise ValueError("empty training or validation loader")
    return {
        "loss": total_loss / samples,
        "bce_loss": total_bce / samples,
        "patient_weighted_bce_loss": total_weighted_bce / samples,
        "smooth_ap_loss": total_ap_loss / samples,
        "labels": np.concatenate(labels) if labels else None,
        "logits": np.concatenate(logits_all) if logits_all else None,
    }


def train_neural(
    data,
    train_indices,
    validation_indices,
    candidate,
    config,
    directory,
    signature,
    progress,
    resume=True,
):
    torch.manual_seed(config["seed"])
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = resolve_device(config["device"])
    train_hrv, validation_hrv, processor_state = preprocess(
        data, train_indices, validation_indices, candidate
    )
    patient_weighting_state = None
    train_sample_weights = None
    if uses_patient_equal_loss(candidate):
        train_sample_weights, patient_weighting_state = patient_equal_weights(
            data.subject_ids,
            train_indices,
        )
    loader_generator = torch.Generator().manual_seed(config["seed"])
    augmentation_generator = torch.Generator().manual_seed(config["seed"] + 2_000_003)
    train_view = FoldView(
        data,
        train_indices,
        train_hrv,
        candidate,
        sample_weights=train_sample_weights,
    )
    train_loader = DataLoader(
        train_view,
        batch_size=config["batch_size"],
        shuffle=True,
        generator=loader_generator,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    validation_loader = DataLoader(
        FoldView(data, validation_indices, validation_hrv, candidate),
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    model = build_sprint_model(candidate, config["dropout"]).to(device)
    if uses_icentia_pretraining(candidate):
        load_pretrained_encoder(model, config["encoder_checkpoint"])
    if uses_afpdb_pretraining(candidate):
        load_afpdb_representation(model, config["afpdb_checkpoint"])
        optimizer = torch.optim.AdamW(
            [
                {
                    "params": list(model.ecg_encoder.parameters())
                    + list(model.temporal.parameters()),
                    "lr": config["afpdb_representation_learning_rate"],
                },
                {"params": model.classifier.parameters(), "lr": config["learning_rate"]},
            ],
            weight_decay=config["weight_decay"],
        )
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"]
        )
    training_criterion = (
        BinaryFocalLoss(gamma=2.0).to(device) if uses_focal_loss(candidate) else None
    )
    latest = directory / "latest.pt"
    history, best_state = [], None
    best_ap, best_epoch, stale, start_epoch = -1.0, 0, 0, 1
    schedule = (
        curriculum_schedule(config["epochs"])
        if uses_horizon_curriculum(candidate)
        else (20,) * config["epochs"]
    )
    if uses_horizon_curriculum(candidate) and data.horizon_labels is None:
        raise ValueError("horizon curriculum requires verified manifest timing labels")
    if resume and latest.is_file():
        saved = torch.load(latest, map_location="cpu", weights_only=True)
        if saved["signature"] != signature:
            raise ValueError("resume configuration/data mismatch; use a new output directory")
        model.load_state_dict(saved["model_state_dict"])
        optimizer.load_state_dict(saved["optimizer_state_dict"])
        for state in optimizer.state.values():
            for name, value in state.items():
                if torch.is_tensor(value):
                    state[name] = value.to(device)
        history, best_state = saved["history"], saved["best_state_dict"]
        best_ap, best_epoch, stale = saved["best_ap"], saved["best_epoch"], saved["stale"]
        start_epoch = saved["epoch"] + 1
        loader_generator.set_state(saved["loader_rng"])
        augmentation_generator.set_state(saved["augmentation_rng"])
        torch.set_rng_state(saved["torch_rng"])
        if device.type == "cuda" and saved["cuda_rng"]:
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
        progress.emit(f"RESUME from completed epoch {saved['epoch']}")

    for epoch in range(start_epoch, config["epochs"] + 1):
        horizon = schedule[epoch - 1]
        if horizon == 20 and stale >= config["patience"]:
            break
        epoch_start = time.monotonic()
        progress.update(epoch=epoch, training_horizon_minutes=horizon)
        if uses_horizon_curriculum(candidate):
            train_view.labels = data.horizon_labels[horizon]
        trained = run_epoch(
            model,
            train_loader,
            device,
            progress,
            optimizer,
            max_batches=config.get("max_train_batches"),
            mask_ecg=uses_ecg_masking(candidate),
            augmentation_generator=augmentation_generator,
            training_criterion=training_criterion,
            patient_equal_loss=uses_patient_equal_loss(candidate),
            ap_loss_coefficient=(
                config.get("smooth_ap_coefficient", SMOOTH_AP_COEFFICIENT)
                if uses_ap_aligned_loss(candidate)
                else 0.0
            ),
            ap_loss_temperature=config.get(
                "smooth_ap_temperature", SMOOTH_AP_TEMPERATURE
            ),
        )
        if horizon == 20:
            predicted = run_epoch(model, validation_loader, device, progress)
            metrics = ranking_metrics(predicted["labels"], expit(predicted["logits"]))
            ap = metrics["pr_auc"]
            if ap is None:
                raise ValueError("inner validation must contain both classes")
            if ap > best_ap:
                best_ap, best_epoch, stale = ap, epoch, 0
                best_state = cpu_state(model)
            else:
                stale += 1
            validation_loss = predicted["loss"]
            validation_auroc = metrics["auroc"]
        else:
            ap = None
            validation_loss = None
            validation_auroc = None
        history.append(
            {
                "epoch": epoch,
                "training_horizon_minutes": horizon,
                "training_positive_windows": int(
                    (train_view.labels[train_indices] == 1).sum()
                ),
                "train_loss": trained["loss"],
                "train_bce_loss": trained["bce_loss"],
                "train_patient_weighted_bce_loss": trained[
                    "patient_weighted_bce_loss"
                ],
                "train_smooth_ap_loss": trained["smooth_ap_loss"],
                "validation_loss": validation_loss,
                "validation_ap": ap,
                "validation_auroc": validation_auroc,
                "epoch_seconds": time.monotonic() - epoch_start,
            }
        )
        # Persist optimizer/RNG/early-stopping state every completed epoch.
        atomic_torch(
            latest,
            {
                "signature": signature,
                "epoch": epoch,
                "history": history,
                "model_state_dict": cpu_state(model),
                "best_state_dict": best_state,
                "optimizer_state_dict": optimizer.state_dict(),
                "best_ap": best_ap,
                "best_epoch": best_epoch,
                "stale": stale,
                "loader_rng": loader_generator.get_state(),
                "augmentation_rng": augmentation_generator.get_state(),
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
            },
        )
        validation_text = (
            f"val_AP={ap:.4f} val_AUROC={validation_auroc:.4f}; "
            if horizon == 20
            else "validation=deferred_until_20min_stage; "
        )
        progress.emit(
            f"EPOCH {epoch:03d}: training_horizon={horizon}min "
            f"train_objective={trained['loss']:.4f} train_BCE={trained['bce_loss']:.4f} "
            f"train_patient_BCE={trained['patient_weighted_bce_loss']:.4f} "
            f"train_smooth_AP={trained['smooth_ap_loss']:.4f} "
            f"{validation_text}best_epoch={best_epoch}; stale={stale}/{config['patience']}; "
            f"epoch_time={history[-1]['epoch_seconds']:.1f}s"
        )
    if best_state is None:
        raise RuntimeError("no valid checkpoint")
    model.load_state_dict(best_state)
    predicted = run_epoch(model, validation_loader, device, progress)
    checkpoint_path = directory / "best.pt"
    atomic_torch(
        checkpoint_path,
        {
            "format": VERSION,
            "candidate": candidate,
            "signature": signature,
            "config": config,
            "model_state_dict": best_state,
            "hrv_preprocessor": processor_state,
            "patient_weighting": patient_weighting_state,
            "best_epoch": best_epoch,
            "train_indices": train_indices.tolist(),
            "validation_indices": validation_indices.tolist(),
        },
    )
    return predicted, {
        "best_epoch": best_epoch,
        "history": history,
        "epochs_completed": len(history),
        "reached_epoch_budget": len(history) == config["epochs"],
        "parameter_count": model.parameter_count,
        "training_recipe": candidate_recipe(candidate),
        "patient_weighting": patient_weighting_state,
        "curriculum_training_positive_counts": (
            {
                str(horizon): int(data.horizon_labels[horizon][train_indices].sum())
                for horizon in (5, 10, 15, 20)
            }
            if uses_horizon_curriculum(candidate)
            else None
        ),
        "checkpoint": checkpoint_path.name,
    }


def describe_predictions(data, indices, predicted, directory):
    y, logits = np.asarray(predicted["labels"]), np.asarray(predicted["logits"], dtype=float)
    if not np.array_equal(y, data.labels[indices]):
        raise AssertionError("prediction order does not match the validation manifest")
    p = expit(logits)
    threshold = select_f1_threshold(y, p)["threshold"]
    metrics = classification_metrics(y, p, threshold)
    metrics.update(ranking_metrics(y, p))
    metrics["brier"] = float(np.mean((p - y) ** 2))
    metrics["nll"] = float(np.logaddexp(0, np.where(y == 1, -logits, logits)).mean())
    metrics["positive_prevalence"] = float(y.mean())
    records, databases = {}, {}
    for position, index in enumerate(indices):
        ref = data.refs[int(index)]
        records.setdefault(ref.subject_id, []).append(position)
        databases.setdefault(ref.subject_id.split(":")[0], []).append(position)
    within = [ranking_metrics(y[pos], p[pos]) for pos in records.values()]
    available_auroc = [row["auroc"] for row in within if row["auroc"] is not None]
    available_ap = [row["pr_auc"] for row in within if row["pr_auc"] is not None]
    metrics["within_record_auroc_mean"] = (
        float(np.mean(available_auroc)) if available_auroc else None
    )
    metrics["within_record_ap_mean"] = float(np.mean(available_ap)) if available_ap else None
    metrics["records_with_both_classes"] = len(available_auroc)
    database_metrics = {
        name: {
            **classification_metrics(y[pos], p[pos], threshold),
            **ranking_metrics(y[pos], p[pos]),
        }
        for name, pos in databases.items()
    }
    path = directory / "validation_predictions.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "dataset_index",
                "subject_id",
                "start_sample_250hz",
                "label",
                "logit",
                "probability",
                "prediction",
                "threshold",
            ]
        )
        for index, label, logit, probability in zip(indices, y, logits, p):
            ref = data.refs[int(index)]
            writer.writerow(
                [
                    int(index),
                    ref.subject_id,
                    ref.start_sample_250hz,
                    int(label),
                    float(logit),
                    float(probability),
                    int(probability >= threshold),
                    threshold,
                ]
            )
    return metrics, database_metrics


def read_oof_predictions(output_dir, candidate, inner_folds):
    rows = []
    for fold in inner_folds:
        path = Path(output_dir) / candidate / f"inner{fold}" / "validation_predictions.csv"
        if not path.is_file():
            return None
        with path.open(newline="", encoding="utf-8") as handle:
            rows.extend(csv.DictReader(handle))
    if not rows:
        return None
    rows.sort(key=lambda row: int(row["dataset_index"]))
    indices = np.asarray([int(row["dataset_index"]) for row in rows], dtype=np.int64)
    if len(np.unique(indices)) != len(indices):
        raise ValueError(f"duplicate OOF dataset index for {candidate}")
    return {
        "indices": indices,
        "subjects": np.asarray([row["subject_id"] for row in rows], dtype=object),
        "labels": np.asarray([int(row["label"]) for row in rows], dtype=np.uint8),
        "probabilities": np.asarray(
            [float(row["probability"]) for row in rows],
            dtype=np.float64,
        ),
    }


def _confidence_interval(values):
    finite = np.asarray([value for value in values if np.isfinite(value)], dtype=np.float64)
    if len(finite) == 0:
        return None
    return [float(np.quantile(finite, 0.025)), float(np.quantile(finite, 0.975))]


def aggregate_oof_predictions(output_dir, candidates, inner_folds, seed, replicates=1000):
    """Summarize pooled OOF predictions with patient-cluster bootstrap intervals."""
    available = {}
    for candidate in candidates:
        predictions = read_oof_predictions(output_dir, candidate, inner_folds)
        if predictions is not None:
            available[candidate] = predictions
    if not available:
        return {}

    reference_name = "paper_control" if "paper_control" in available else next(iter(available))
    reference = available[reference_name]
    for candidate, predictions in available.items():
        if not np.array_equal(predictions["indices"], reference["indices"]):
            raise ValueError(f"OOF indices differ between {reference_name} and {candidate}")
        if not np.array_equal(predictions["labels"], reference["labels"]):
            raise ValueError(f"OOF labels differ between {reference_name} and {candidate}")
        if not np.array_equal(predictions["subjects"], reference["subjects"]):
            raise ValueError(f"OOF subjects differ between {reference_name} and {candidate}")

    subjects = reference["subjects"]
    unique_subjects = np.unique(subjects)
    positions = {
        subject: np.flatnonzero(subjects == subject)
        for subject in unique_subjects
    }
    rng = np.random.default_rng(seed + 3_000_009)
    draws = [
        rng.choice(unique_subjects, size=len(unique_subjects), replace=True)
        for _ in range(replicates)
    ]

    output = {}
    control_bootstrap_ap = None
    for candidate, predictions in available.items():
        labels = predictions["labels"]
        probabilities = predictions["probabilities"]
        pooled = ranking_metrics(labels, probabilities)
        per_record = [
            ranking_metrics(labels[index], probabilities[index])
            for index in positions.values()
        ]
        record_auroc = [row["auroc"] for row in per_record if row["auroc"] is not None]
        record_ap = [row["pr_auc"] for row in per_record if row["pr_auc"] is not None]
        bootstrap_ap = []
        bootstrap_auroc = []
        for draw in draws:
            sampled = np.concatenate([positions[subject] for subject in draw])
            metrics = ranking_metrics(labels[sampled], probabilities[sampled])
            if metrics["pr_auc"] is not None:
                bootstrap_ap.append(metrics["pr_auc"])
                bootstrap_auroc.append(metrics["auroc"])
        if candidate == "paper_control":
            control_bootstrap_ap = np.asarray(bootstrap_ap, dtype=np.float64)
        output[candidate] = {
            "windows": int(len(labels)),
            "records": int(len(unique_subjects)),
            "positive_windows": int(labels.sum()),
            "pooled_AP": pooled["pr_auc"],
            "pooled_AUROC": pooled["auroc"],
            "within_record_AP_mean": float(np.mean(record_ap)) if record_ap else None,
            "within_record_AUROC_mean": (
                float(np.mean(record_auroc)) if record_auroc else None
            ),
            "records_with_both_classes": len(record_auroc),
            "cluster_bootstrap_replicates": len(bootstrap_ap),
            "pooled_AP_cluster_95pct_CI": _confidence_interval(bootstrap_ap),
            "pooled_AUROC_cluster_95pct_CI": _confidence_interval(bootstrap_auroc),
            "_bootstrap_ap": np.asarray(bootstrap_ap, dtype=np.float64),
        }

    if control_bootstrap_ap is not None:
        for candidate, summary in output.items():
            values = summary.pop("_bootstrap_ap")
            if len(values) == len(control_bootstrap_ap):
                summary["paired_AP_delta_vs_control_cluster_95pct_CI"] = (
                    _confidence_interval(values - control_bootstrap_ap)
                )
    else:
        for summary in output.values():
            summary.pop("_bootstrap_ap")
    return output


def aggregate_results(rows, candidates, inner_folds, smoke=False):
    ranking = []

    def fold_metric(candidate, metric):
        return {
            row["inner_fold"]: row["validation_metrics"].get(metric)
            for row in rows
            if row["candidate"] == candidate
        }

    controls = fold_metric("paper_control", "pr_auc")
    control_within = fold_metric("paper_control", "within_record_auroc_mean")
    for name in candidates:
        selected = [row for row in rows if row["candidate"] == name]
        values = [row["validation_metrics"]["pr_auc"] for row in selected]
        if not values:
            continue
        paired_control = [
            row["validation_metrics"]["pr_auc"] - controls[row["inner_fold"]]
            for row in selected
            if row["inner_fold"] in controls
        ]
        paired_within = [
            row["validation_metrics"]["within_record_auroc_mean"]
            - control_within[row["inner_fold"]]
            for row in selected
            if row["inner_fold"] in control_within
            and row["validation_metrics"]["within_record_auroc_mean"] is not None
            and control_within[row["inner_fold"]] is not None
        ]
        parent = PARENT_CANDIDATE[name]
        parent_values = fold_metric(parent, "pr_auc") if parent else {}
        parent_within = fold_metric(parent, "within_record_auroc_mean") if parent else {}
        paired_parent = [
            row["validation_metrics"]["pr_auc"] - parent_values[row["inner_fold"]]
            for row in selected
            if row["inner_fold"] in parent_values
        ]
        paired_parent_within = [
            row["validation_metrics"]["within_record_auroc_mean"]
            - parent_within[row["inner_fold"]]
            for row in selected
            if row["inner_fold"] in parent_within
            and row["validation_metrics"]["within_record_auroc_mean"] is not None
            and parent_within[row["inner_fold"]] is not None
        ]
        complete = len(selected) == len(inner_folds)
        full = complete and sorted(inner_folds) == list(range(5)) and not smoke
        mean_control_delta = float(np.mean(paired_control)) if paired_control else None
        control_wins = sum(delta > 0 for delta in paired_control)
        within_delta = float(np.mean(paired_within)) if paired_within else None
        mean_parent_delta = float(np.mean(paired_parent)) if paired_parent else None
        parent_wins = sum(delta > 0 for delta in paired_parent)
        parent_within_delta = (
            float(np.mean(paired_parent_within)) if paired_parent_within else None
        )
        gate_deltas = paired_parent if parent else paired_control
        gate_mean = mean_parent_delta if parent else mean_control_delta
        gate_wins = parent_wins if parent else control_wins
        gate_within = parent_within_delta if parent else within_delta
        within_auroc_values = [
            row["validation_metrics"]["within_record_auroc_mean"]
            for row in selected
            if row["validation_metrics"]["within_record_auroc_mean"] is not None
        ]
        within_ap_values = [
            row["validation_metrics"]["within_record_ap_mean"]
            for row in selected
            if row["validation_metrics"]["within_record_ap_mean"] is not None
        ]
        ranking.append(
            {
                "candidate": name,
                "parent_candidate": parent,
                "completed_folds": len(selected),
                "complete": complete,
                "validation_AP_mean": float(np.mean(values)),
                "validation_AP_std": float(np.std(values, ddof=1)) if len(values) > 1 else None,
                "validation_AUROC_mean": float(
                    np.mean([r["validation_metrics"]["auroc"] for r in selected])
                ),
                "within_record_AUROC_mean": (
                    float(np.mean(within_auroc_values)) if within_auroc_values else None
                ),
                "within_record_AP_mean": (
                    float(np.mean(within_ap_values)) if within_ap_values else None
                ),
                "paired_AP_delta_vs_control": mean_control_delta,
                "paired_wins_vs_control": control_wins,
                "paired_within_record_AUROC_delta_vs_control": within_delta,
                "paired_AP_delta_vs_parent": mean_parent_delta,
                "paired_wins_vs_parent": parent_wins,
                "paired_within_record_AUROC_delta_vs_parent": parent_within_delta,
                "parameter_count": selected[0]["parameter_count"],
                "folds_reaching_epoch_budget": sum(r["reached_epoch_budget"] for r in selected),
                "promising_for_second_seed": bool(
                    full
                    and len(gate_deltas) == 5
                    and gate_mean is not None
                    and gate_mean > 0.01
                    and gate_wins >= 3
                    and gate_within is not None
                    and gate_within >= 0.0
                ),
            }
        )
    ranking.sort(
        key=lambda row: (not row["complete"], -row["validation_AP_mean"], row["parameter_count"])
    )

    factorial_effects = {}
    comparisons = {
        "ap_alignment_under_natural_masked_bce": (
            "se_last_mask_ap",
            "se_last_mask",
        ),
        "patient_equal_weighting_under_masked_bce": (
            "se_last_mask_patient_equal",
            "se_last_mask",
        ),
        "ap_alignment_under_patient_equal_weighting": (
            "se_last_mask_patient_equal_ap",
            "se_last_mask_patient_equal",
        ),
        "horizon_curriculum_under_masked_bce": (
            "se_last_mask_curriculum",
            "se_last_mask",
        ),
        "afpdb_sequence_transfer_under_masked_bce": (
            "se_last_mask_afpdb",
            "se_last_mask",
        ),
        "icentia_initialization_under_masked_bce": (
            "se_last_mask_icentia",
            "se_last_mask",
        ),
        "masking_under_bce": ("se_last_mask", "se_last_natural"),
        "focal_without_mask": ("se_last_focal", "se_last_natural"),
        "masking_under_focal": ("se_last_focal_mask", "se_last_focal"),
        "focal_with_mask": ("se_last_focal_mask", "se_last_mask"),
    }
    for effect, (treatment, reference) in comparisons.items():
        treatment_values = fold_metric(treatment, "pr_auc")
        reference_values = fold_metric(reference, "pr_auc")
        common = sorted(set(treatment_values) & set(reference_values))
        deltas = [treatment_values[fold] - reference_values[fold] for fold in common]
        factorial_effects[effect] = {
            "treatment": treatment,
            "reference": reference,
            "paired_folds": common,
            "AP_delta_mean": float(np.mean(deltas)) if deltas else None,
            "wins": sum(delta > 0 for delta in deltas),
        }
    base = fold_metric("se_last_natural", "pr_auc")
    masked = fold_metric("se_last_mask", "pr_auc")
    focal = fold_metric("se_last_focal", "pr_auc")
    combined = fold_metric("se_last_focal_mask", "pr_auc")
    interaction_folds = sorted(set(base) & set(masked) & set(focal) & set(combined))
    interactions = [
        (combined[fold] - focal[fold]) - (masked[fold] - base[fold])
        for fold in interaction_folds
    ]
    factorial_effects["masking_by_focal_interaction"] = {
        "paired_folds": interaction_folds,
        "AP_interaction_mean": float(np.mean(interactions)) if interactions else None,
    }
    return {
        "format": VERSION,
        "scope": "development selection only; no new outer-test inference",
        "outer_test_iterated": False,
        "smoke_test": smoke,
        "inner_folds": inner_folds,
        "ranking": ranking,
        "factorial_effects": factorial_effects,
        "all_jobs_complete": len(rows) == len(candidates) * len(inner_folds),
        "selection_rule": (
            "mean paired inner-validation average precision under equal folds and "
            "budget; each candidate is compared with its pre-specified parent and "
            "within-record AUROC is a no-regression guardrail"
        ),
        "confirmation_rule": (
            "AP gain versus the candidate's parent >0.01, wins in >=3/5 folds, "
            "and non-negative paired within-record AUROC delta nominate a "
            "second-seed confirmation; not a significance test"
        ),
        "honesty_note": (
            "Earlier outer-test results were already viewed. This is exploratory "
            "development, not an untouched final evaluation."
        ),
        "threshold_note": (
            "F1 thresholds fitted on validation are diagnostics, not independent "
            "performance estimates."
        ),
        "folds": rows,
    }


def run(args, progress):
    base = CacheWindowDataset(args.cache_root, args.manifest, ecg_normalization="segment_zscore")
    needs_curriculum = any(uses_horizon_curriculum(model) for model in args.models)
    development, excluded = make_outer_fold_indices(
        base.labels, base.subject_ids, 5, args.outer_fold
    )
    horizon_labels = (
        build_horizon_labels(
            args.manifest,
            base.window_refs,
            allowed_indices=development,
        )
        if needs_curriculum
        else None
    )
    if len(set(base.subject_ids[development])) < 5:
        raise ValueError("five inner folds require at least five development records")
    progress.update(phase="loading_development_hrv")
    progress.emit(
        f"DEVELOPMENT windows={len(development)}; excluded outer-test windows={len(excluded)}"
    )
    data = DevelopmentData(base, development, horizon_labels=horizon_labels)
    source_hash = hashlib.sha256()
    for source in sorted(Path(__file__).parent.glob("*.py")):
        source_hash.update(source.name.encode())
        source_hash.update(source.read_bytes())
    cache_hash = hashlib.sha256()
    for subject in sorted(set(base.subject_ids[development])):
        cache_hash.update(subject.encode())
        cache_hash.update(
            (args.cache_root / subject.replace(":", "__") / "metadata.json").read_bytes()
        )
    config = {
        "epochs": args.epochs,
        "patience": args.patience,
        "batch_size": args.batch_size,
        "learning_rate": 5e-4,
        "weight_decay": 1e-4,
        "dropout": 0.3,
        "seed": args.seed,
        "device": str(resolve_device(args.device)),
        "outer_fold": args.outer_fold,
        "n_splits": 5,
        "sampling": "natural",
        "loss": "candidate_specific_bce_focal_or_ap_aligned",
        "ecg_normalization": "segment_zscore",
        "max_train_batches": args.max_train_batches,
        "cluster_bootstrap_replicates": args.bootstrap_replicates,
        "encoder_checkpoint": (
            str(args.encoder_checkpoint.resolve())
            if args.encoder_checkpoint is not None
            else None
        ),
        "encoder_checkpoint_sha256": (
            digest_file(args.encoder_checkpoint)
            if args.encoder_checkpoint is not None
            else None
        ),
        "afpdb_checkpoint": (
            str(args.afpdb_checkpoint.resolve())
            if args.afpdb_checkpoint is not None
            else None
        ),
        "afpdb_checkpoint_sha256": (
            digest_file(args.afpdb_checkpoint)
            if args.afpdb_checkpoint is not None
            else None
        ),
        "afpdb_representation_learning_rate": 1e-4,
        "smooth_ap_coefficient": SMOOTH_AP_COEFFICIENT,
        "smooth_ap_temperature": SMOOTH_AP_TEMPERATURE,
        "horizon_curriculum_schedule": (
            list(curriculum_schedule(args.epochs))
            if needs_curriculum
            else None
        ),
        "manifest_sha256": digest_file(args.manifest),
        "source_sha256": source_hash.hexdigest(),
        "cache_metadata_sha256": cache_hash.hexdigest(),
        "torch_version": str(torch.__version__),
        "sklearn_version": sklearn.__version__,
    }
    experiment_path = args.output_dir / "experiment.json"
    experiment = {
        "format": VERSION,
        "config": config,
        "candidates": args.models,
        "candidate_recipes": {
            candidate: candidate_recipe(candidate) for candidate in args.models
        },
        "inner_folds": args.inner_folds,
        "development_indices": development.tolist(),
        "excluded_outer_indices": excluded.tolist(),
        "group_identity_caveat": (
            "database:record identifiers; cross-record patient identity needs "
            "source verification"
        ),
    }
    if experiment_path.is_file():
        previous = json.loads(experiment_path.read_text())
        if previous != experiment:
            raise ValueError(
                "experiment plan, data, source, or software changed; use a new output directory"
            )
    atomic_json(experiment_path, experiment)
    # Verify all planned splits before spending any training time.
    splits = {}
    for fold in args.inner_folds:
        train, validation = make_validation_split(development, base.subject_ids, 5, fold)
        data.check(train)
        data.check(validation)
        if set(base.subject_ids[train]) & set(base.subject_ids[validation]):
            raise AssertionError("record leakage in inner split")
        if any(len(np.unique(base.labels[part])) != 2 for part in [train, validation]):
            raise ValueError(f"inner fold {fold} lacks a class; do not silently change the split")
        splits[fold] = (train, validation)
    rows, durations = [], []
    total = len(args.models) * len(args.inner_folds)
    # Complete the cheap baselines early, then compare every heavy candidate.
    for candidate in args.models:
        for fold in args.inner_folds:
            job_start = time.monotonic()
            directory = args.output_dir / candidate / f"inner{fold}"
            directory.mkdir(parents=True, exist_ok=True)
            signature = hashlib.sha256(
                json.dumps(
                    {**config, "candidate": candidate, "inner_fold": fold}, sort_keys=True
                ).encode()
            ).hexdigest()
            summary_path = directory / "summary.json"
            progress.update(
                job=f"{candidate}/inner{fold}", phase="starting", epoch=0, batch=0, batches=0
            )
            progress.emit(f"START {len(rows) + 1}/{total}: {candidate}, inner fold {fold}")
            if args.resume and summary_path.is_file():
                summary = json.loads(summary_path.read_text())
                if summary.get("signature") != signature:
                    raise ValueError(f"resume mismatch for {directory}; use a new output directory")
                for name, checksum in summary["artifact_sha256"].items():
                    if (
                        not (directory / name).is_file()
                        or digest_file(directory / name) != checksum
                    ):
                        raise ValueError(f"incomplete or changed artifact: {directory / name}")
                progress.emit("RESUME: completed job verified and skipped")
            else:
                train, validation = splits[fold]
                predictions, extra = train_neural(
                    data,
                    train,
                    validation,
                    candidate,
                    config,
                    directory,
                    signature,
                    progress,
                    resume=args.resume,
                )
                metrics, db_metrics = describe_predictions(data, validation, predictions, directory)
                summary = {
                    "candidate": candidate,
                    "inner_fold": fold,
                    "signature": signature,
                    "config": config,
                    **extra,
                    "validation_metrics": metrics,
                    "validation_by_database": db_metrics,
                    "train_windows": len(train),
                    "validation_windows": len(validation),
                    "train_records": len(set(base.subject_ids[train])),
                    "validation_records": len(set(base.subject_ids[validation])),
                    "outer_test_iterated": False,
                    "elapsed_seconds": time.monotonic() - job_start,
                    "artifact_sha256": {
                        name: digest_file(directory / name)
                        for name in (extra["checkpoint"], "validation_predictions.csv")
                    },
                }
                atomic_json(summary_path, summary)
                durations.append((candidate, summary["elapsed_seconds"]))
            rows.append(summary)
            report = aggregate_results(
                rows, args.models, args.inner_folds, bool(args.max_train_batches)
            )
            atomic_json(args.output_dir / "sprint_summary.json", report)
            progress.emit(
                f"DONE {len(rows)}/{total}: {candidate}, inner={fold}; "
                f"val_AP={summary['validation_metrics']['pr_auc']:.4f}; "
                f"elapsed={summary['elapsed_seconds'] / 60:.1f} min"
            )
            comparable = [seconds for name, seconds in durations if name == candidate]
            if comparable:
                remaining = len(args.inner_folds) - sum(r["candidate"] == candidate for r in rows)
                progress.emit(
                    f"Observed-rate ETA for remaining {candidate} folds: "
                    f"~{np.mean(comparable) * remaining / 60:.1f} min; early stopping and I/O vary"
                )
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    data.check(list(data.accessed))
    if set(data.accessed) & set(map(int, excluded)):
        raise AssertionError("outer-test signal access detected")
    report["oof_development_metrics"] = aggregate_oof_predictions(
        args.output_dir,
        args.models,
        args.inner_folds,
        seed=args.seed,
        replicates=args.bootstrap_replicates,
    )
    atomic_json(args.output_dir / "oof_summary.json", report["oof_development_metrics"])
    atomic_json(args.output_dir / "sprint_summary.json", report)
    progress.emit("FINISHED: validation-only comparison; no outer-test predictions produced")
    print(
        json.dumps({key: value for key, value in report.items() if key != "folds"}, indent=2),
        flush=True,
    )
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--models", nargs="+", choices=CANDIDATES, default=list(CANDIDATES))
    parser.add_argument("--inner-folds", nargs="+", type=int, default=list(range(5)))
    parser.add_argument("--outer-fold", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cuda")
    parser.add_argument("--max-train-batches", type=int, default=None, help="SMOKE TEST ONLY")
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    parser.add_argument(
        "--encoder-checkpoint",
        type=Path,
        help="Required when se_last_mask_icentia is selected",
    )
    parser.add_argument(
        "--afpdb-checkpoint",
        type=Path,
        help="Required when se_last_mask_afpdb is selected",
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.epochs < 1 or args.patience < 1 or args.batch_size < 1:
        parser.error("epochs, patience, and batch size must be positive")
    if any(uses_horizon_curriculum(model) for model in args.models) and args.epochs < 7:
        parser.error("se_last_mask_curriculum requires at least 7 epochs")
    if args.max_train_batches is not None and args.max_train_batches < 1:
        parser.error("max-train-batches must be positive")
    if args.bootstrap_replicates < 100:
        parser.error("bootstrap-replicates must be at least 100")
    if len(set(args.models)) != len(args.models):
        parser.error("models must be unique")
    if len(set(args.inner_folds)) != len(args.inner_folds) or any(
        f not in range(5) for f in args.inner_folds
    ):
        parser.error("inner folds must be unique integers 0..4")
    if args.outer_fold not in range(5):
        parser.error("outer-fold must be 0..4")
    needs_encoder = any(uses_icentia_pretraining(model) for model in args.models)
    needs_afpdb = any(uses_afpdb_pretraining(model) for model in args.models)
    if needs_encoder and args.encoder_checkpoint is None:
        parser.error("--encoder-checkpoint is required for se_last_mask_icentia")
    if args.encoder_checkpoint is not None and not args.encoder_checkpoint.is_file():
        parser.error(f"encoder checkpoint does not exist: {args.encoder_checkpoint}")
    if args.encoder_checkpoint is not None and not needs_encoder:
        parser.error(
            "--encoder-checkpoint was supplied but no Icentia-pretrained model was selected"
        )
    if needs_afpdb and args.afpdb_checkpoint is None:
        parser.error("--afpdb-checkpoint is required for se_last_mask_afpdb")
    if args.afpdb_checkpoint is not None and not args.afpdb_checkpoint.is_file():
        parser.error(f"AFPDB checkpoint does not exist: {args.afpdb_checkpoint}")
    if args.afpdb_checkpoint is not None and not needs_afpdb:
        parser.error(
            "--afpdb-checkpoint was supplied but se_last_mask_afpdb was not selected"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # Runtime-local lock is automatically released if the process is killed.
    lock_name = hashlib.sha256(str(args.output_dir.resolve()).encode()).hexdigest()[:20]
    with Path(f"/tmp/afib_improve_{lock_name}.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(
                "this output directory already has an active runner in this runtime"
            ) from error
        with Progress(args.output_dir) as progress:
            run(args, progress)


if __name__ == "__main__":
    main()

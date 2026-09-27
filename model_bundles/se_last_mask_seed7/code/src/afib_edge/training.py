"""Leakage-safe training utilities for the AF prediction models."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

from .dataset import CacheWindowDataset
from .hrv_preprocess import HRVFoldPreprocessor
from .models import ModelConfig, build_model


@dataclass(frozen=True)
class TrainingConfig:
    """Training defaults aligned with the reference paper."""

    model_name: str = "proposed"
    batch_size: int = 32
    epochs: int = 50
    learning_rate: float = 5e-4
    weight_decay: float = 1e-4
    focal_gamma: float = 2.0
    dropout: float = 0.30
    patience: int = 10
    gradient_clip: float = 1.0
    n_splits: int = 5
    validation_fold: int = 0
    num_workers: int = 0
    seed: int = 7
    device: str = "auto"
    max_train_batches: int | None = None
    max_eval_batches: int | None = None
    ecg_normalization: str = "none"
    sampling: str = "balanced"
    loss: str = "focal"

    def validate(self) -> None:
        if self.batch_size <= 0 or self.epochs <= 0:
            raise ValueError("batch_size and epochs must be positive")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("learning_rate must be positive and weight_decay non-negative")
        if self.focal_gamma < 0 or self.patience < 0:
            raise ValueError("focal_gamma and patience must be non-negative")
        if self.gradient_clip <= 0 or self.n_splits < 2:
            raise ValueError("gradient_clip must be positive and n_splits at least two")
        if self.num_workers < 0:
            raise ValueError("num_workers cannot be negative")
        if self.max_train_batches is not None and self.max_train_batches <= 0:
            raise ValueError("max_train_batches must be positive when provided")
        if self.max_eval_batches is not None and self.max_eval_batches <= 0:
            raise ValueError("max_eval_batches must be positive when provided")
        if self.ecg_normalization not in {"none", "segment_zscore"}:
            raise ValueError("ecg_normalization must be none or segment_zscore")
        if self.sampling not in {"balanced", "natural"}:
            raise ValueError("sampling must be balanced or natural")
        if self.loss not in {"focal", "bce"}:
            raise ValueError("loss must be focal or bce")


class BinaryFocalLoss(nn.Module):
    """Binary focal loss used with a weighted sampler for class imbalance."""

    def __init__(self, gamma: float = 2.0, alpha: float | None = None) -> None:
        super().__init__()
        if gamma < 0:
            raise ValueError("gamma must be non-negative")
        if alpha is not None and not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1]")
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        targets = targets.to(dtype=logits.dtype)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p_t = torch.exp(-bce)
        loss = (1.0 - p_t).pow(self.gamma) * bce
        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
            loss = alpha_t * loss
        return loss.mean()


def resolve_device(requested: str) -> torch.device:
    """Resolve ``auto`` to CUDA when available, otherwise CPU."""
    normalized = requested.strip().lower()
    if normalized == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if normalized not in {"cpu", "cuda"}:
        raise ValueError("device must be auto, cpu, or cuda")
    if normalized == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(normalized)


def make_outer_fold_indices(
    labels: np.ndarray,
    subject_ids: np.ndarray,
    n_splits: int = 5,
    fold: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return patient-disjoint outer train/test indices."""
    y = np.asarray(labels)
    groups = np.asarray(subject_ids)
    if y.ndim != 1 or groups.ndim != 1 or len(y) != len(groups):
        raise ValueError("labels and subject_ids must be one-dimensional and equally sized")
    if len(np.unique(groups)) < n_splits:
        raise ValueError("there are fewer unique patients than requested folds")
    if not 0 <= fold < n_splits:
        raise ValueError(f"fold must be in [0, {n_splits})")

    splitter = GroupKFold(n_splits=n_splits)
    splits = list(splitter.split(np.zeros(len(y)), y, groups))
    train_indices, test_indices = splits[fold]
    if set(groups[train_indices]) & set(groups[test_indices]):
        raise AssertionError("patient leakage detected in outer split")
    return train_indices.astype(np.int64), test_indices.astype(np.int64)


def make_validation_split(
    train_indices: np.ndarray,
    subject_ids: np.ndarray,
    n_splits: int = 5,
    validation_fold: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Split outer-training patients into disjoint inner train/validation sets."""
    outer_train = np.asarray(train_indices, dtype=np.int64)
    groups = np.asarray(subject_ids)
    unique_groups = np.unique(groups[outer_train])
    effective_splits = min(n_splits, len(unique_groups))
    if effective_splits < 2:
        raise ValueError("at least two training patients are required for validation")
    if not 0 <= validation_fold < effective_splits:
        raise ValueError(f"validation_fold must be in [0, {effective_splits})")

    splitter = GroupKFold(n_splits=effective_splits)
    relative_train, relative_validation = list(
        splitter.split(np.zeros(len(outer_train)), groups=groups[outer_train])
    )[validation_fold]
    inner_train = outer_train[relative_train]
    validation = outer_train[relative_validation]
    if set(groups[inner_train]) & set(groups[validation]):
        raise AssertionError("patient leakage detected in validation split")
    return inner_train, validation


def make_weighted_sampler(labels: Iterable[int], seed: int = 7) -> WeightedRandomSampler:
    """Sample inverse-frequency weighted windows with replacement."""
    y = np.asarray(list(labels), dtype=np.int64)
    if y.ndim != 1 or len(y) == 0 or np.any((y < 0) | (y > 1)):
        raise ValueError("labels must be a non-empty binary vector")
    counts = np.bincount(y, minlength=2).astype(np.float64)
    weights = np.ones(len(y), dtype=np.float64)
    present = counts > 0
    for class_id in (0, 1):
        if present[class_id]:
            weights[y == class_id] = 1.0 / counts[class_id]
    generator = torch.Generator()
    generator.manual_seed(seed)
    return WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(y),
        replacement=True,
        generator=generator,
    )


def classification_metrics(
    labels: np.ndarray, probabilities: np.ndarray, threshold: float = 0.5
) -> dict[str, float]:
    """Compute thresholded binary metrics without hiding zero-denominator cases."""
    y_true = np.asarray(labels, dtype=np.uint8)
    y_prob = np.asarray(probabilities, dtype=np.float64)
    if y_true.ndim != 1 or y_prob.ndim != 1 or len(y_true) != len(y_prob) or len(y_true) == 0:
        raise ValueError("labels and probabilities must be non-empty equal-length vectors")
    if np.any((y_true < 0) | (y_true > 1)):
        raise ValueError("labels must be binary")
    if not np.all(np.isfinite(y_prob)) or np.any((y_prob < 0.0) | (y_prob > 1.0)):
        raise ValueError("probabilities must be finite and in [0, 1]")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")
    y_pred = (y_prob >= threshold).astype(np.uint8)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "threshold": float(threshold),
        "accuracy": (tp + tn) / len(y_true),
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "positive_count": float(np.sum(y_true)),
        "sample_count": float(len(y_true)),
        "true_positives": float(tp),
        "true_negatives": float(tn),
        "false_positives": float(fp),
        "false_negatives": float(fn),
    }


def ranking_metrics(
    labels: np.ndarray, probabilities: np.ndarray
) -> dict[str, float | None]:
    """Compute threshold-free metrics, or ``None`` for a one-class smoke subset."""
    y_true = np.asarray(labels, dtype=np.uint8)
    y_prob = np.asarray(probabilities, dtype=np.float64)
    if y_true.ndim != 1 or y_prob.ndim != 1 or len(y_true) != len(y_prob) or len(y_true) == 0:
        raise ValueError("labels and probabilities must be non-empty equal-length vectors")
    if not np.all(np.isfinite(y_prob)) or np.any((y_prob < 0.0) | (y_prob > 1.0)):
        raise ValueError("probabilities must be finite and in [0, 1]")
    if len(np.unique(y_true)) != 2:
        return {"auroc": None, "pr_auc": None}
    return {
        "auroc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
    }


def select_f1_threshold(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    """Select an F1-optimal threshold using validation predictions only.

    Exact observed probabilities are evaluated. If several thresholds have the
    same F1, the highest threshold is retained to prefer fewer false positives.
    """
    y_true = np.asarray(labels, dtype=np.uint8)
    y_prob = np.asarray(probabilities, dtype=np.float64)
    if len(np.unique(y_true)) != 2:
        raise ValueError("threshold selection requires both validation classes")
    ranking_metrics(y_true, y_prob)
    candidates = np.unique(y_prob)
    best_threshold = float(candidates[0])
    best_f1 = -1.0
    for threshold in candidates:
        f1 = classification_metrics(y_true, y_prob, float(threshold))["f1"]
        if f1 > best_f1 or (np.isclose(f1, best_f1) and threshold > best_threshold):
            best_f1 = f1
            best_threshold = float(threshold)
    return {"threshold": best_threshold, "f1": float(best_f1)}


def _move_batch(batch: dict[str, Tensor | str], device: torch.device) -> tuple[Tensor, Tensor, Tensor]:
    return (
        batch["ecg"].to(device, non_blocking=True),
        batch["hrv"].to(device, non_blocking=True),
        batch["label"].to(device, non_blocking=True),
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    gradient_clip: float = 1.0,
    max_batches: int | None = None,
) -> float:
    model.train()
    total_loss = 0.0
    total_samples = 0
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        ecg, hrv, labels = _move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(ecg, hrv), labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        optimizer.step()
        batch_size = labels.shape[0]
        total_loss += float(loss.detach().cpu()) * batch_size
        total_samples += batch_size
    if total_samples == 0:
        raise ValueError("training loader produced no batches")
    return total_loss / total_samples


@torch.no_grad()
def predict_loader(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    max_batches: int | None = None,
) -> dict[str, object]:
    model.eval()
    losses: list[float] = []
    labels: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        ecg, hrv, batch_labels = _move_batch(batch, device)
        logits = model(ecg, hrv)
        losses.append(float(criterion(logits, batch_labels).cpu()) * len(batch_labels))
        labels.append(batch_labels.cpu().numpy().astype(np.uint8))
        probabilities.append(torch.sigmoid(logits).cpu().numpy())
    if not labels:
        raise ValueError("evaluation loader produced no batches")
    return {
        "labels": np.concatenate(labels),
        "probabilities": np.concatenate(probabilities),
        "loss": float(sum(losses) / sum(len(y) for y in labels)),
    }


def metrics_from_predictions(
    predictions: dict[str, object], threshold: float = 0.5
) -> dict[str, float | None]:
    labels = np.asarray(predictions["labels"])
    probabilities = np.asarray(predictions["probabilities"])
    metrics = classification_metrics(labels, probabilities, threshold=threshold)
    metrics.update(ranking_metrics(labels, probabilities))
    metrics["loss"] = float(predictions["loss"])
    return metrics


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    max_batches: int | None = None,
    threshold: float = 0.5,
) -> dict[str, float | None]:
    predictions = predict_loader(model, loader, criterion, device, max_batches=max_batches)
    return metrics_from_predictions(predictions, threshold=threshold)


def train_one_fold(
    dataset: CacheWindowDataset,
    config: TrainingConfig,
    fold: int = 0,
    verbose: bool = True,
    evaluate_outer_test: bool = True,
) -> dict[str, object]:
    """Train one outer fold with an inner patient-wise validation split."""
    config.validate()
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    device = resolve_device(config.device)
    labels = dataset.labels
    subject_ids = dataset.subject_ids
    outer_train, outer_test = make_outer_fold_indices(
        labels, subject_ids, n_splits=config.n_splits, fold=fold
    )
    inner_train, validation = make_validation_split(
        outer_train,
        subject_ids,
        n_splits=config.n_splits,
        validation_fold=config.validation_fold,
    )

    hrv_preprocessor = HRVFoldPreprocessor().fit(dataset.raw_hrv_for_indices(inner_train))
    dataset.hrv_preprocessor = hrv_preprocessor
    dataset.ecg_normalization = config.ecg_normalization
    train_subset = Subset(dataset, inner_train.tolist())
    validation_subset = Subset(dataset, validation.tolist())
    test_subset = Subset(dataset, outer_test.tolist())
    sampler = (
        make_weighted_sampler(labels[inner_train], seed=config.seed)
        if config.sampling == "balanced"
        else None
    )
    loader_kwargs = {
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(
        train_subset,
        sampler=sampler,
        shuffle=sampler is None,
        **loader_kwargs,
    )
    validation_loader = DataLoader(validation_subset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_subset, shuffle=False, **loader_kwargs)

    model = build_model(config.model_name, ModelConfig(dropout=config.dropout)).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    criterion: nn.Module = (
        BinaryFocalLoss(gamma=config.focal_gamma)
        if config.loss == "focal"
        else nn.BCEWithLogitsLoss()
    )
    best_state: dict[str, Tensor] | None = None
    best_validation_score = -float("inf")
    selection_metric = "validation_pr_auc"
    best_epoch = 0
    epochs_without_improvement = 0
    history: list[dict[str, object]] = []

    for epoch in range(1, config.epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            gradient_clip=config.gradient_clip,
            max_batches=config.max_train_batches,
        )
        validation_metrics = evaluate(
            model,
            validation_loader,
            criterion,
            device,
            max_batches=config.max_eval_batches,
        )
        record = {"epoch": epoch, "train_loss": train_loss, "validation": validation_metrics}
        history.append(record)
        if verbose:
            pr_auc = validation_metrics["pr_auc"]
            pr_auc_text = "n/a" if pr_auc is None else f"{pr_auc:.4f}"
            print(
                f"epoch {epoch:03d}: train_loss={train_loss:.4f} "
                f"val_loss={validation_metrics['loss']:.4f} "
                f"val_f1@0.5={validation_metrics['f1']:.4f} "
                f"val_pr_auc={pr_auc_text}",
                flush=True,
            )
        if validation_metrics["pr_auc"] is None:
            current_score = -float(validation_metrics["loss"])
            selection_metric = "validation_loss_smoke_fallback"
        else:
            current_score = float(validation_metrics["pr_auc"])
        if current_score > best_validation_score:
            best_validation_score = current_score
            best_epoch = epoch
            best_state = {
                name: parameter.detach().cpu().clone()
                for name, parameter in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.patience:
                break

    if best_state is None:
        raise RuntimeError("no validation checkpoint was produced")
    model.load_state_dict(best_state)
    model.to(device)
    validation_predictions = predict_loader(
        model, validation_loader, criterion, device, max_batches=config.max_eval_batches
    )
    validation_labels = np.asarray(validation_predictions["labels"])
    if len(np.unique(validation_labels)) == 2:
        threshold_selection = select_f1_threshold(
            validation_labels, np.asarray(validation_predictions["probabilities"])
        )
        selected_threshold = threshold_selection["threshold"]
    else:
        selected_threshold = 0.5
    validation_metrics = metrics_from_predictions(
        validation_predictions, threshold=selected_threshold
    )
    if evaluate_outer_test:
        test_predictions = predict_loader(
            model, test_loader, criterion, device, max_batches=config.max_eval_batches
        )
        test_metrics = metrics_from_predictions(test_predictions, threshold=selected_threshold)
        test_metrics_at_0_5 = metrics_from_predictions(test_predictions, threshold=0.5)
    else:
        test_predictions = None
        test_metrics = None
        test_metrics_at_0_5 = None
    return {
        "model": model,
        "hrv_preprocessor": hrv_preprocessor,
        "history": history,
        "selection_metric": selection_metric,
        "selected_threshold": selected_threshold,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "test_metrics_at_0_5": test_metrics_at_0_5,
        "validation_predictions": validation_predictions,
        "test_predictions": test_predictions,
        "device": str(device),
        "best_epoch": best_epoch,
        "parameter_count": int(model.parameter_count),
        "inner_train_indices": inner_train,
        "validation_indices": validation,
        "outer_test_indices": outer_test,
        "outer_train_indices": outer_train,
    }


__all__ = [
    "BinaryFocalLoss",
    "TrainingConfig",
    "classification_metrics",
    "evaluate",
    "metrics_from_predictions",
    "make_outer_fold_indices",
    "make_validation_split",
    "make_weighted_sampler",
    "predict_loader",
    "ranking_metrics",
    "resolve_device",
    "select_f1_threshold",
    "train_one_epoch",
    "train_one_fold",
]

"""Training-only patient balancing and AP-aligned ranking objectives."""

from __future__ import annotations

from collections import Counter

import numpy as np
import torch


SMOOTH_AP_TEMPERATURE = 0.1
SMOOTH_AP_COEFFICIENT = 0.2


def patient_equal_weights(
    subject_ids: np.ndarray,
    train_indices: np.ndarray,
) -> tuple[np.ndarray, dict[str, object]]:
    """Give every inner-training patient equal total BCE weight.

    All training windows remain present and are still shuffled naturally. The
    returned weights have mean one over the authorized training indices, so the
    optimizer's loss scale remains comparable with ordinary BCE. Non-training
    indices receive zero and cannot contribute to the objective.
    """
    subjects = np.asarray(subject_ids, dtype=object)
    indices = np.asarray(train_indices, dtype=np.int64)
    if subjects.ndim != 1 or indices.ndim != 1 or len(indices) == 0:
        raise ValueError("subjects and non-empty training indices must be one-dimensional")
    if len(np.unique(indices)) != len(indices):
        raise ValueError("training indices must be unique")
    if indices.min() < 0 or indices.max() >= len(subjects):
        raise ValueError("training index lies outside the subject vector")

    counts = Counter(map(str, subjects[indices]))
    if len(counts) < 2:
        raise ValueError("patient-equal weighting requires at least two training patients")
    scale = len(indices) / len(counts)
    weights = np.zeros(len(subjects), dtype=np.float32)
    for index in indices:
        weights[index] = scale / counts[str(subjects[index])]

    selected = weights[indices]
    if not np.isfinite(selected).all() or np.any(selected <= 0):
        raise FloatingPointError("patient weights must be positive and finite")
    if not np.isclose(float(selected.mean()), 1.0, rtol=1e-6, atol=1e-6):
        raise AssertionError("patient weights were not normalized to mean one")
    totals = {
        subject: float(weights[indices[subjects[indices] == subject]].sum())
        for subject in counts
    }
    expected_total = scale
    if not all(
        np.isclose(total, expected_total, rtol=1e-6, atol=1e-6)
        for total in totals.values()
    ):
        raise AssertionError("training patients do not have equal total weight")

    state = {
        "kind": "inverse_training_window_count_per_patient",
        "normalization": "mean_training_window_weight_equals_one",
        "training_windows": int(len(indices)),
        "training_patients": int(len(counts)),
        "minimum_windows_per_patient": int(min(counts.values())),
        "maximum_windows_per_patient": int(max(counts.values())),
        "minimum_window_weight": float(selected.min()),
        "maximum_window_weight": float(selected.max()),
        "equal_total_weight_per_patient": float(expected_total),
    }
    return weights, state


def smooth_binary_average_precision_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    temperature: float = SMOOTH_AP_TEMPERATURE,
    sample_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Differentiable within-batch approximation of binary AP.

    The sigmoid relaxes pairwise rank indicators. Batches without both classes
    return a differentiable zero, leaving patient-weighted BCE as the objective.
    """
    if logits.ndim != 1 or targets.ndim != 1 or logits.shape != targets.shape:
        raise ValueError("Smooth-AP expects equal one-dimensional logits and targets")
    if temperature <= 0:
        raise ValueError("Smooth-AP temperature must be positive")
    weights = torch.ones_like(logits) if sample_weights is None else sample_weights
    if weights.shape != logits.shape:
        raise ValueError("Smooth-AP weights must match the logits")
    if not torch.isfinite(weights).all() or bool(torch.any(weights <= 0)):
        raise ValueError("Smooth-AP weights must be positive and finite")
    positive = targets > 0.5
    negative = ~positive
    if not bool(positive.any()) or not bool(negative.any()):
        return logits.sum() * 0.0

    positive_logits = logits[positive]
    positive_weights = weights[positive]
    # Row i compares every score j with positive score i. Adding 0.5 while
    # retaining the sigmoid self-comparison (also 0.5) implements
    # 1 + sum_{j != i} I(score_j > score_i) smoothly.
    all_differences = logits.unsqueeze(0) - positive_logits.unsqueeze(1)
    positive_differences = positive_logits.unsqueeze(0) - positive_logits.unsqueeze(1)
    smooth_rank = 0.5 * positive_weights + (
        torch.sigmoid(all_differences / temperature) * weights.unsqueeze(0)
    ).sum(dim=1)
    smooth_positive_rank = 0.5 * positive_weights + (
        torch.sigmoid(positive_differences / temperature)
        * positive_weights.unsqueeze(0)
    ).sum(dim=1)
    precision = smooth_positive_rank / smooth_rank
    smooth_ap = (precision * positive_weights).sum() / positive_weights.sum()
    return 1.0 - smooth_ap


__all__ = [
    "SMOOTH_AP_COEFFICIENT",
    "SMOOTH_AP_TEMPERATURE",
    "patient_equal_weights",
    "smooth_binary_average_precision_loss",
]

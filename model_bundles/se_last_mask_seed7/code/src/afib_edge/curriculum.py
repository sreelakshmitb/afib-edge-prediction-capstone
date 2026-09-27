"""Leakage-safe training labels and schedule for a horizon curriculum.

The cached ECG/HRV tensors are unchanged. Only inner-training labels are
temporarily narrowed to closer AF-onset horizons. Validation and every final
metric retain the original 20-minute target.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .dataset import WindowRef


CURRICULUM_HORIZONS_MINUTES = (5, 10, 15, 20)
CURRICULUM_WARMUP_EPOCHS_PER_HORIZON = 2


def build_horizon_labels(
    manifest_path: Path,
    refs: tuple[WindowRef, ...],
    *,
    fs_hz: int = 250,
    allowed_indices: np.ndarray | None = None,
) -> dict[int, np.ndarray]:
    """Derive nested horizon labels only for the authorized development rows."""
    grouped: dict[str, list[dict[str, str]]] = {}
    with Path(manifest_path).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            grouped.setdefault(row["subject_id"], []).append(row)
    for subject_rows in grouped.values():
        subject_rows.sort(key=lambda row: int(row["start_sample_250hz"]))
    grouped = dict(sorted(grouped.items()))
    rows = [row for subject_rows in grouped.values() for row in subject_rows]
    if len(rows) != len(refs):
        raise ValueError("manifest/reference length mismatch for horizon labels")
    if allowed_indices is None:
        allowed = np.arange(len(rows), dtype=np.int64)
    else:
        allowed = np.asarray(allowed_indices, dtype=np.int64)
        if allowed.ndim != 1 or len(np.unique(allowed)) != len(allowed):
            raise ValueError("allowed horizon-label indices must be unique and one-dimensional")
        if len(allowed) == 0 or allowed.min() < 0 or allowed.max() >= len(rows):
            raise ValueError("allowed horizon-label index is outside the manifest")
    allowed_set = set(map(int, allowed))

    labels = {
        horizon: np.zeros(len(rows), dtype=np.uint8)
        for horizon in CURRICULUM_HORIZONS_MINUTES
    }
    for index, (row, ref) in enumerate(zip(rows, refs, strict=True)):
        if row["subject_id"] != ref.subject_id:
            raise ValueError(f"manifest/reference subject mismatch at row {index}")
        if int(row["start_sample_250hz"]) != ref.start_sample_250hz:
            raise ValueError(f"manifest/reference start mismatch at row {index}")
        original = int(row["label"])
        if original not in (0, 1) or original != ref.label:
            raise ValueError(f"manifest/reference label mismatch at row {index}")
        if index not in allowed_set:
            continue
        if "end_sample_250hz" not in row or "future_af_onset_sample_250hz" not in row:
            raise ValueError("manifest lacks end/onset timing required by the curriculum")

        onset_text = row["future_af_onset_sample_250hz"].strip()
        if original == 0:
            if onset_text:
                raise ValueError("negative 20-minute row unexpectedly carries a future onset")
            continue
        if not onset_text:
            raise ValueError("positive 20-minute row is missing its future onset")
        end = int(row["end_sample_250hz"])
        onset = int(onset_text)
        gap = onset - end
        twenty_minutes = 20 * 60 * fs_hz
        if gap < 0 or gap >= twenty_minutes:
            raise ValueError("positive row onset lies outside the fixed 20-minute horizon")
        for horizon in CURRICULUM_HORIZONS_MINUTES:
            labels[horizon][index] = int(gap < horizon * 60 * fs_hz)

    original = np.asarray([ref.label for ref in refs], dtype=np.uint8)
    if not np.array_equal(labels[20][allowed], original[allowed]):
        raise AssertionError("derived 20-minute labels differ from the locked target")
    for shorter, longer in zip(
        CURRICULUM_HORIZONS_MINUTES[:-1],
        CURRICULUM_HORIZONS_MINUTES[1:],
        strict=True,
    ):
        if np.any(labels[shorter][allowed] > labels[longer][allowed]):
            raise AssertionError("horizon labels are not nested")
    return labels


def curriculum_schedule(total_epochs: int) -> tuple[int, ...]:
    """Return a fixed equal-budget 5→10→15→20 minute schedule."""
    warmup = CURRICULUM_WARMUP_EPOCHS_PER_HORIZON
    minimum = warmup * 3 + 1
    if total_epochs < minimum:
        raise ValueError(f"horizon curriculum requires at least {minimum} epochs")
    return (
        (5,) * warmup
        + (10,) * warmup
        + (15,) * warmup
        + (20,) * (total_epochs - 3 * warmup)
    )


__all__ = [
    "CURRICULUM_HORIZONS_MINUTES",
    "CURRICULUM_WARMUP_EPOCHS_PER_HORIZON",
    "build_horizon_labels",
    "curriculum_schedule",
]

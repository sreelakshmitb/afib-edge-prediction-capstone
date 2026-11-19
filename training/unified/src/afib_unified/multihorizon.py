"""Leakage-safe multi-horizon targets and discrete-time survival loss."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .manifest import sha256_file


HORIZONS_MINUTES: tuple[int, ...] = (4, 8, 12, 16, 20)
LEAD_BINS: tuple[str, ...] = ("00_04", "04_08", "08_12", "12_16", "16_20")
JOIN_KEYS: tuple[str, ...] = (
    "source_family", "record_id", "subject_id", "start_sample_250hz"
)
TIMING_COLUMNS: tuple[str, ...] = (
    "observation_start_seconds",
    "forecast_start_seconds",
    "forecast_end_seconds",
    "end_sample_250hz",
    "horizon_end_sample_250hz",
    "af_onset_seconds",
    "lead_time_minutes",
    "next_af_onset_minutes",
    "minutes_since_af_end",
)


def lead_bin_index(lead_time_minutes: np.ndarray | pd.Series) -> np.ndarray:
    """Map exact positive lead times to right-closed four-minute intervals."""
    lead = np.asarray(lead_time_minutes, dtype=np.float64)
    if not np.isfinite(lead).all() or np.any(lead <= 0.0) or np.any(lead > 20.0):
        raise ValueError("positive lead times must be finite and in (0, 20]")
    return np.searchsorted(np.asarray(HORIZONS_MINUTES), lead, side="left").astype(np.int8)


def cumulative_targets(labels: np.ndarray, event_bins: np.ndarray) -> np.ndarray:
    """Return binary onset-by-horizon targets with shape ``[N,5]``."""
    labels = np.asarray(labels, dtype=np.int8)
    event_bins = np.asarray(event_bins, dtype=np.int8)
    if labels.shape != event_bins.shape:
        raise ValueError("labels and event bins must have the same shape")
    if not np.isin(labels, (0, 1)).all():
        raise ValueError("labels must be binary")
    if np.any((labels == 0) & (event_bins != -1)):
        raise ValueError("censored negatives must use event_bin=-1")
    if np.any((labels == 1) & ((event_bins < 0) | (event_bins >= 5))):
        raise ValueError("positive event bins must be in [0,4]")
    horizons = np.arange(5, dtype=np.int8)[None, :]
    return ((labels[:, None] == 1) & (event_bins[:, None] <= horizons)).astype(np.uint8)


def build_multihorizon_manifest(
    cached_manifest_path: Path, combined_manifest_path: Path
) -> pd.DataFrame:
    """Attach exact onset timing to the immutable cache-addressable manifest."""
    cached = pd.read_csv(cached_manifest_path).sort_values("manifest_index").reset_index(drop=True)
    combined = pd.read_csv(combined_manifest_path)
    required_cached = {
        "manifest_index", "label", "lead_bin", "event_id", "window_id",
        "outer_test_fold", *JOIN_KEYS,
    }
    required_combined = {"label", "lead_bin", "event_id", *JOIN_KEYS, *TIMING_COLUMNS}
    if missing := required_cached - set(cached.columns):
        raise ValueError(f"cached manifest missing columns: {sorted(missing)}")
    if missing := required_combined - set(combined.columns):
        raise ValueError(f"combined manifest missing columns: {sorted(missing)}")
    if not np.array_equal(cached.manifest_index.to_numpy(), np.arange(len(cached))):
        raise ValueError("cached manifest_index must equal row position")
    if cached.window_id.duplicated().any():
        raise ValueError("cached manifest contains duplicate window IDs")
    if combined.duplicated(list(JOIN_KEYS)).any():
        duplicates = int(combined.duplicated(list(JOIN_KEYS), keep=False).sum())
        raise ValueError(f"combined timing keys are not unique: {duplicates} rows")

    timing = combined[[*JOIN_KEYS, "label", "lead_bin", "event_id", *TIMING_COLUMNS]].copy()
    timing = timing.rename(
        columns={
            "label": "timing_label",
            "lead_bin": "timing_lead_bin",
            "event_id": "timing_event_id",
        }
    )
    output = cached.merge(timing, on=list(JOIN_KEYS), how="left", validate="one_to_one")
    if len(output) != len(cached) or output.observation_start_seconds.isna().any():
        raise ValueError("not every cached window has one exact timing match")
    if not output.label.astype(int).eq(output.timing_label.astype(int)).all():
        raise ValueError("binary labels disagree between manifests")

    positive = output.label.astype(int).eq(1)
    cached_events = output.event_id.fillna("").astype(str)
    timing_events = output.timing_event_id.fillna("").astype(str)
    if not cached_events[positive].eq(timing_events[positive]).all():
        raise ValueError("positive event IDs disagree between manifests")
    cached_bins = output.lead_bin.fillna("").astype(str)
    timing_bins = output.timing_lead_bin.fillna("").astype(str)
    if not cached_bins[positive].eq(timing_bins[positive]).all():
        raise ValueError("positive lead bins disagree between manifests")

    output["event_bin"] = -1
    event_indices = lead_bin_index(output.loc[positive, "lead_time_minutes"])
    output.loc[positive, "event_bin"] = event_indices
    output["event_bin"] = output.event_bin.astype(np.int8)
    reconstructed = pd.Series(
        [LEAD_BINS[int(index)] for index in event_indices], index=output.index[positive]
    )
    if not reconstructed.eq(output.loc[positive, "lead_bin"].astype(str)).all():
        raise ValueError("exact lead times do not reconstruct the declared lead bins")
    if output.loc[~positive, "lead_time_minutes"].notna().any():
        raise ValueError("negative windows unexpectedly contain positive lead times")
    unsafe_negative = (
        ~positive
        & output.next_af_onset_minutes.notna()
        & output.next_af_onset_minutes.le(20.0)
    )
    if unsafe_negative.any():
        raise ValueError("negative windows contain AF onset inside the 20-minute horizon")

    targets = cumulative_targets(output.label.to_numpy(), output.event_bin.to_numpy())
    for column, values in zip(
        ("target_04", "target_08", "target_12", "target_16", "target_20"),
        targets.T,
    ):
        output[column] = values
    output = output.drop(columns=["timing_label", "timing_lead_bin", "timing_event_id"])
    return output.sort_values("manifest_index").reset_index(drop=True)


def build_and_save_multihorizon_targets(
    cached_manifest_path: Path, combined_manifest_path: Path, output_dir: Path
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "multihorizon_cached_manifest.csv"
    manifest = build_multihorizon_manifest(cached_manifest_path, combined_manifest_path)
    manifest.to_csv(output_path, index=False)
    positive = manifest.label.astype(int).eq(1)
    gates = {
        "all_8292_cached_windows_preserved": len(manifest) == 8292,
        "all_1054_positive_windows_preserved": int(positive.sum()) == 1054,
        "all_positive_lead_times_exact": bool(
            manifest.loc[positive, "lead_time_minutes"].notna().all()
        ),
        "all_five_event_bins_present": set(manifest.loc[positive, "event_bin"]) == set(range(5)),
        "twenty_minute_target_matches_binary_label": bool(
            manifest.target_20.astype(int).eq(manifest.label.astype(int)).all()
        ),
        "cumulative_targets_monotonic": bool(
            np.diff(
                manifest[["target_04", "target_08", "target_12", "target_16", "target_20"]]
                .to_numpy(dtype=np.int8), axis=1
            ).min(initial=0) >= 0
        ),
        "no_negative_onset_inside_horizon": not bool((
            manifest.loc[~positive, "next_af_onset_minutes"].notna()
            & manifest.loc[~positive, "next_af_onset_minutes"].le(20.0)
        ).any()),
        "no_duplicate_windows": not bool(manifest.window_id.duplicated().any()),
        "outer_fold_assignments_preserved": bool(
            manifest.groupby("subject_id").outer_test_fold.nunique().eq(1).all()
        ),
    }
    event_bin_counts = {
        LEAD_BINS[int(key)]: int(value)
        for key, value in manifest.loc[positive, "event_bin"].value_counts().sort_index().items()
    }
    summary = {
        "format": "afib_multihorizon_target_manifest_v1",
        "windows": int(len(manifest)),
        "positives": int(positive.sum()),
        "negatives": int((~positive).sum()),
        "patients": int(manifest.subject_id.nunique()),
        "positive_events": int(manifest.loc[positive, "event_id"].nunique()),
        "horizons_minutes": list(HORIZONS_MINUTES),
        "event_bin_counts": event_bin_counts,
        "source_by_label": {
            f"label_{int(label)}:{source}": int(len(group))
            for (label, source), group in manifest.groupby(["label", "source_family"], sort=True)
        },
        "cached_manifest_sha256": sha256_file(cached_manifest_path),
        "combined_manifest_sha256": sha256_file(combined_manifest_path),
        "multihorizon_manifest_sha256": sha256_file(output_path),
        "outer_test_evaluated": False,
        "gates": gates,
        "all_gates_pass": all(gates.values()),
    }
    (output_dir / "target_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    if not summary["all_gates_pass"]:
        raise RuntimeError("multi-horizon target integrity gate failed")
    return summary


def cumulative_risk_from_logits(hazard_logits: Any) -> Any:
    """Convert conditional hazard logits to monotonic cumulative risk."""
    import torch

    if hazard_logits.ndim != 2 or hazard_logits.shape[1] != 5:
        raise ValueError("hazard logits must have shape [B,5]")
    return 1.0 - torch.cumprod(1.0 - torch.sigmoid(hazard_logits), dim=1)


def balanced_discrete_time_nll(hazard_logits: Any, event_bin: Any) -> Any:
    """Class-balanced, per-window-normalized discrete-time survival loss.

    Positives contribute survival through intervals preceding their event and
    an event term in their interval. Censored negatives contribute survival
    through all five intervals. Each window is normalized by its number of
    observed intervals, then positive and negative window means receive equal
    weight. This prevents the five negative survival terms and the natural 7:1
    class imbalance from overwhelming the relatively scarce onset examples.
    """
    import torch
    from torch.nn import functional as F

    if hazard_logits.ndim != 2 or hazard_logits.shape[1] != 5:
        raise ValueError("hazard logits must have shape [B,5]")
    bins = event_bin.to(device=hazard_logits.device, dtype=torch.long)
    if bins.ndim != 1 or len(bins) != len(hazard_logits):
        raise ValueError("event_bin must have shape [B]")
    if torch.any((bins < -1) | (bins >= 5)):
        raise ValueError("event_bin values must be -1 or 0..4")
    interval = torch.arange(5, device=hazard_logits.device)[None, :]
    positive = bins >= 0
    observed = torch.where(positive[:, None], interval <= bins[:, None], True)
    event = positive[:, None] & (interval == bins[:, None])
    log_term = torch.where(event, F.logsigmoid(hazard_logits), F.logsigmoid(-hazard_logits))
    per_window = -(log_term * observed).sum(dim=1) / observed.sum(dim=1).clamp_min(1)
    if positive.any() and (~positive).any():
        return 0.5 * per_window[positive].mean() + 0.5 * per_window[~positive].mean()
    return per_window.mean()


__all__ = [
    "HORIZONS_MINUTES",
    "JOIN_KEYS",
    "LEAD_BINS",
    "balanced_discrete_time_nll",
    "build_and_save_multihorizon_targets",
    "build_multihorizon_manifest",
    "cumulative_risk_from_logits",
    "cumulative_targets",
    "lead_bin_index",
]

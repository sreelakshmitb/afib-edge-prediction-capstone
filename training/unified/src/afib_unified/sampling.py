"""Hierarchical batch planning that prevents majority and patient dominance."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .contracts import LEAD_BINS


@dataclass(frozen=True)
class HierarchicalSamplerConfig:
    """Fixed v1 sampling policy for the AF forecasting training set."""

    batch_size: int = 32
    positives_per_batch: int = 8
    positive_draws_per_patient: int = 2
    batches_per_epoch: int | None = None
    iridia_source_fraction: float = 0.5
    near_negative_fraction: float = 0.2
    medium_negative_fraction: float = 0.3
    far_negative_fraction: float = 0.5
    positive_lead_bins: tuple[str, ...] = tuple(LEAD_BINS)
    stratified_negative_sources: tuple[str, ...] = ("iridia",)
    seed: int = 7

    def validate(self) -> None:
        if self.batch_size <= 1:
            raise ValueError("batch_size must be greater than one")
        if not 0 < self.positives_per_batch < self.batch_size:
            raise ValueError("positives_per_batch must be between zero and batch_size")
        if self.positive_draws_per_patient <= 0:
            raise ValueError("positive_draws_per_patient must be positive")
        if self.batches_per_epoch is not None and self.batches_per_epoch <= 0:
            raise ValueError("batches_per_epoch must be positive when provided")
        if not 0.0 <= self.iridia_source_fraction <= 1.0:
            raise ValueError("iridia_source_fraction must be in [0, 1]")
        negative_sum = (
            self.near_negative_fraction
            + self.medium_negative_fraction
            + self.far_negative_fraction
        )
        if not np.isclose(negative_sum, 1.0):
            raise ValueError("negative stratum fractions must sum to one")
        if min(
            self.near_negative_fraction,
            self.medium_negative_fraction,
            self.far_negative_fraction,
        ) < 0.0:
            raise ValueError("negative stratum fractions cannot be negative")
        if not self.positive_lead_bins:
            raise ValueError("positive_lead_bins cannot be empty")
        if not set(self.positive_lead_bins).issubset(set(LEAD_BINS)):
            raise ValueError("positive_lead_bins contains an unknown lead bin")
        if len(set(self.positive_lead_bins)) != len(self.positive_lead_bins):
            raise ValueError("positive_lead_bins cannot contain duplicates")
        if not self.stratified_negative_sources:
            raise ValueError("stratified_negative_sources cannot be empty")
        if not set(self.stratified_negative_sources).issubset({"iridia", "legacy"}):
            raise ValueError("unknown stratified negative source")


def _allocate(total: int, weights: dict[str, float], rng: np.random.Generator) -> dict[str, int]:
    usable = {str(key): float(value) for key, value in weights.items() if value > 0.0}
    if total < 0 or not usable:
        raise ValueError("allocation requires a non-negative total and positive weights")
    scale = sum(usable.values())
    raw = {key: total * value / scale for key, value in usable.items()}
    result = {key: int(math.floor(value)) for key, value in raw.items()}
    remaining = total - sum(result.values())
    if remaining:
        candidates = list(usable)
        rng.shuffle(candidates)
        candidates.sort(key=lambda key: raw[key] - result[key], reverse=True)
        for key in candidates[:remaining]:
            result[key] += 1
    if sum(result.values()) != total:
        raise AssertionError("allocation failed to conserve total")
    return result


class _Selector:
    def __init__(
        self,
        manifest: pd.DataFrame,
        rng: np.random.Generator,
        epoch: int,
        positive_lead_bins: tuple[str, ...],
    ) -> None:
        self.manifest = manifest
        self.rng = rng
        self.epoch = epoch
        self.positive_lead_bins = positive_lead_bins
        self.patient_counts: Counter = Counter()
        self.event_counts: Counter = Counter()
        self.lead_counts: Counter = Counter()
        self.positive_draw_number = 0

    def positive(self, family: str, used: set[int]) -> int:
        candidates = self.manifest[
            self.manifest["training_eligible"].astype(bool)
            & self.manifest["label"].eq(1)
            & self.manifest["source_family"].astype(str).eq(family)
            & self.manifest["lead_bin"].astype(str).isin(self.positive_lead_bins)
        ]
        candidates = candidates[~candidates.index.isin(used)]
        if candidates.empty:
            raise ValueError(
                f"no unused eligible positive windows remain in this batch for {family}"
            )

        target_bin = self.positive_lead_bins[
            (self.positive_draw_number + self.epoch) % len(self.positive_lead_bins)
        ]
        self.positive_draw_number += 1
        patients = sorted(candidates["subject_id"].astype(str).unique())
        patient_scores = np.asarray(
            [self.patient_counts[(family, "positive", patient)] for patient in patients]
        )
        tied_patients = [
            patient
            for patient, score in zip(patients, patient_scores)
            if score == patient_scores.min()
        ]
        target_patients = set(
            candidates.loc[candidates["lead_bin"].astype(str).eq(target_bin), "subject_id"]
            .astype(str)
            .unique()
        )
        preferred_patients = [patient for patient in tied_patients if patient in target_patients]
        selected_patient = str(
            self.rng.choice(preferred_patients if preferred_patients else tied_patients)
        )
        self.patient_counts[(family, "positive", selected_patient)] += 1
        patient_rows = candidates[candidates["subject_id"].astype(str).eq(selected_patient)]

        available_bins = sorted(patient_rows["lead_bin"].astype(str).unique())
        minimum_bin_count = min(self.lead_counts[lead_bin] for lead_bin in available_bins)
        tied_bins = [
            lead_bin
            for lead_bin in available_bins
            if self.lead_counts[lead_bin] == minimum_bin_count
        ]
        selected_bin = target_bin if target_bin in tied_bins else str(self.rng.choice(tied_bins))
        self.lead_counts[selected_bin] += 1

        bin_rows = patient_rows[patient_rows["lead_bin"].astype(str).eq(selected_bin)]
        events = sorted(bin_rows["event_id"].astype(str).unique())
        event_scores = np.asarray([self.event_counts[event] for event in events])
        tied_events = [
            event for event, score in zip(events, event_scores) if score == event_scores.min()
        ]
        selected_event = str(self.rng.choice(tied_events))
        self.event_counts[selected_event] += 1
        rows = bin_rows[bin_rows["event_id"].astype(str).eq(selected_event)]
        return int(self.rng.choice(rows.index.to_numpy(dtype=np.int64)))

    def negative(self, family: str, stratum: str | None, used: set[int]) -> int:
        candidates = self.manifest[
            self.manifest["training_eligible"].astype(bool)
            & self.manifest["label"].eq(0)
            & self.manifest["source_family"].astype(str).eq(family)
        ]
        candidates = candidates[~candidates.index.isin(used)]
        if candidates.empty:
            raise ValueError(
                f"no unused eligible negative windows remain in this batch for {family}"
            )
        if stratum is not None:
            preferred = candidates[candidates["negative_stratum"].astype(str).eq(stratum)]
            if not preferred.empty:
                candidates = preferred

        patients = sorted(candidates["subject_id"].astype(str).unique())
        patient_scores = np.asarray(
            [self.patient_counts[(family, "negative", patient)] for patient in patients]
        )
        selected_patient = str(
            self.rng.choice(
                [
                    patient
                    for patient, score in zip(patients, patient_scores)
                    if score == patient_scores.min()
                ]
            )
        )
        self.patient_counts[(family, "negative", selected_patient)] += 1
        rows = candidates[candidates["subject_id"].astype(str).eq(selected_patient)]
        return int(self.rng.choice(rows.index.to_numpy(dtype=np.int64)))


def _family_weights(manifest: pd.DataFrame, label: int, config: HierarchicalSamplerConfig) -> dict[str, float]:
    eligible = manifest[manifest["training_eligible"].astype(bool) & manifest["label"].eq(label)]
    if label == 1:
        eligible = eligible[
            eligible["lead_bin"].astype(str).isin(config.positive_lead_bins)
        ]
    present = set(eligible["source_family"].astype(str).unique())
    if present == {"iridia", "legacy"}:
        return {
            "iridia": config.iridia_source_fraction,
            "legacy": 1.0 - config.iridia_source_fraction,
        }
    if not present:
        raise ValueError(f"no eligible windows for label {label}")
    return {family: 1.0 for family in sorted(present)}


def build_epoch_schedule(
    manifest: pd.DataFrame,
    config: HierarchicalSamplerConfig = HierarchicalSamplerConfig(),
    *,
    epoch: int = 0,
) -> pd.DataFrame:
    """Return an exact-ratio, reproducible batch schedule for one training epoch."""
    config.validate()
    required = {
        "manifest_index",
        "dataset",
        "source_family",
        "subject_id",
        "label",
        "event_id",
        "lead_bin",
        "negative_stratum",
        "training_eligible",
    }
    if missing := required - set(manifest.columns):
        raise ValueError(f"sampling manifest missing columns: {sorted(missing)}")
    if manifest["manifest_index"].duplicated().any():
        raise ValueError("manifest_index must be unique")
    if epoch < 0:
        raise ValueError("epoch cannot be negative")

    eligible = manifest[manifest["training_eligible"].astype(bool)].copy()
    if set(eligible["label"].unique()) != {0, 1}:
        raise ValueError("eligible training manifest must contain both classes")
    eligible = eligible.set_index("manifest_index", drop=False)
    rng = np.random.default_rng(config.seed + epoch * 1009)
    selector = _Selector(eligible, rng, epoch, config.positive_lead_bins)

    positive_patients = int(eligible.loc[eligible["label"].eq(1), "subject_id"].nunique())
    if config.batches_per_epoch is None:
        positive_draws = positive_patients * config.positive_draws_per_patient
        batches = max(1, math.ceil(positive_draws / config.positives_per_batch))
    else:
        batches = config.batches_per_epoch

    positive_family_weights = _family_weights(eligible, 1, config)
    negative_family_weights = _family_weights(eligible, 0, config)
    negative_stratum_weights = {
        "near_20_60": config.near_negative_fraction,
        "medium_60_360": config.medium_negative_fraction,
        "far_360_plus_or_none": config.far_negative_fraction,
    }
    stratum_draw_counts: Counter = Counter()
    rows: list[dict] = []
    for batch_id in range(batches):
        used: set[int] = set()
        indices: list[int] = []
        positive_alloc = _allocate(config.positives_per_batch, positive_family_weights, rng)
        negative_count = config.batch_size - config.positives_per_batch
        negative_alloc = _allocate(negative_count, negative_family_weights, rng)

        for family, count in positive_alloc.items():
            for _ in range(count):
                index = selector.positive(family, used)
                used.add(index)
                indices.append(index)

        for family, count in negative_alloc.items():
            if family in config.stratified_negative_sources:
                for _ in range(count):
                    next_total = sum(
                        stratum_draw_counts[(family, stratum)]
                        for stratum in negative_stratum_weights
                    ) + 1
                    deficits = {
                        stratum: weight * next_total
                        - stratum_draw_counts[(family, stratum)]
                        for stratum, weight in negative_stratum_weights.items()
                    }
                    maximum = max(deficits.values())
                    tied = [
                        stratum
                        for stratum, deficit in deficits.items()
                        if np.isclose(deficit, maximum)
                    ]
                    stratum = str(rng.choice(tied))
                    stratum_draw_counts[(family, stratum)] += 1
                    index = selector.negative(family, stratum, used)
                    used.add(index)
                    indices.append(index)
            else:
                for _ in range(count):
                    index = selector.negative(family, None, used)
                    used.add(index)
                    indices.append(index)

        if len(indices) != config.batch_size:
            raise AssertionError("batch planner emitted the wrong batch size")
        rng.shuffle(indices)
        for slot, index in enumerate(indices):
            source = eligible.loc[index]
            rows.append(
                {
                    "batch_id": batch_id,
                    "slot": slot,
                    "manifest_index": int(index),
                    "label": int(source["label"]),
                    "source_family": str(source["source_family"]),
                    "dataset": str(source["dataset"]),
                    "subject_id": str(source["subject_id"]),
                    "event_id": str(source["event_id"]),
                    "lead_bin": str(source["lead_bin"]),
                    "negative_stratum": str(source["negative_stratum"]),
                }
            )

    schedule = pd.DataFrame(rows)
    per_batch = schedule.groupby("batch_id")["label"].agg(["size", "sum"])
    if not per_batch["size"].eq(config.batch_size).all():
        raise AssertionError("batch size invariant failed")
    if not per_batch["sum"].eq(config.positives_per_batch).all():
        raise AssertionError("positive ratio invariant failed")
    return schedule


def audit_epoch_schedule(schedule: pd.DataFrame) -> dict:
    """Summarize an epoch schedule without treating repeated draws as new evidence."""
    duplicate_slots = int(
        schedule.duplicated(["batch_id", "manifest_index"], keep=False).sum()
    )
    positives = schedule[schedule["label"].eq(1)]
    negatives = schedule[schedule["label"].eq(0)]
    patient_ranges = {}
    for family, group in positives.groupby("source_family"):
        draws = group.groupby("subject_id").size()
        patient_ranges[str(family)] = {
            "min": int(draws.min()),
            "max": int(draws.max()),
            "patients": int(len(draws)),
        }
    return {
        "samples": int(len(schedule)),
        "batches": int(schedule["batch_id"].nunique()),
        "positive_samples": int(len(positives)),
        "negative_samples": int(len(negatives)),
        "sampled_positive_fraction": float(schedule["label"].mean()),
        "duplicate_manifest_indices_within_batch": duplicate_slots,
        "source_by_label": {
            f"label_{int(label)}:{family}": int(count)
            for (label, family), count in schedule.groupby(["label", "source_family"]).size().items()
        },
        "positive_lead_bins": {
            str(key): int(value)
            for key, value in positives["lead_bin"].value_counts().sort_index().items()
        },
        "negative_strata": {
            str(key): int(value)
            for key, value in negatives["negative_stratum"].value_counts().sort_index().items()
        },
        "positive_patient_draw_range": {
            "min": int(positives.groupby("subject_id").size().min()),
            "max": int(positives.groupby("subject_id").size().max()),
        },
        "positive_patient_draw_range_by_source": patient_ranges,
    }


def prior_logit_offset(real_prevalence: float, sampled_prevalence: float = 0.25) -> float:
    """Return the additive logit correction from sampled to real prevalence."""
    if not 0.0 < real_prevalence < 1.0 or not 0.0 < sampled_prevalence < 1.0:
        raise ValueError("prevalences must be strictly between zero and one")
    real_log_odds = math.log(real_prevalence / (1.0 - real_prevalence))
    sampled_log_odds = math.log(sampled_prevalence / (1.0 - sampled_prevalence))
    return float(real_log_odds - sampled_log_odds)


__all__ = [
    "HierarchicalSamplerConfig",
    "audit_epoch_schedule",
    "build_epoch_schedule",
    "prior_logit_offset",
]

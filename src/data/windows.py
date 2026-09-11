"""Causal AF-onset labeling and fixed-shape observation construction."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from data.contracts import (
    MODEL_RATE_HZ,
    OBSERVATION_MINUTES,
    PREDICTION_HORIZON_MINUTES,
    SAMPLES_PER_SEGMENT,
    SEGMENT_SECONDS,
    SEGMENTS_PER_OBSERVATION,
)
from data.types import ObservationDescriptor, RhythmInterval

OBSERVATION_SECONDS = OBSERVATION_MINUTES * 60
PREDICTION_HORIZON_SECONDS = PREDICTION_HORIZON_MINUTES * 60
DEFAULT_WINDOW_STRIDE_SECONDS = int(OBSERVATION_SECONDS * 0.8)  # 20% overlap


def _overlap_seconds(
    first_start: float,
    first_end: float,
    second_start: float,
    second_end: float,
) -> float:
    return max(0.0, min(first_end, second_end) - max(first_start, second_start))


def observation_is_explicit_sinus(
    start_seconds: float,
    end_seconds: float,
    rhythm_intervals: Sequence[RhythmInterval],
) -> bool:
    """Require complete observation coverage by explicit WFDB normal-rhythm '(N' intervals."""
    cursor = start_seconds
    for interval in sorted(rhythm_intervals, key=lambda item: item.start_seconds):
        overlap = _overlap_seconds(
            start_seconds,
            end_seconds,
            interval.start_seconds,
            interval.end_seconds,
        )
        if overlap <= 0:
            continue
        covered_start = max(start_seconds, interval.start_seconds)
        covered_end = min(end_seconds, interval.end_seconds)
        if covered_start > cursor + 1e-9:
            return False
        if interval.rhythm.upper() != "N":
            return False
        cursor = max(cursor, covered_end)
        if cursor >= end_seconds - 1e-9:
            return True
    return False


def generate_observation_descriptors(
    *,
    database: str,
    record_id: str,
    patient_id: str,
    record_duration_seconds: float,
    rhythm_intervals: Sequence[RhythmInterval],
    stride_seconds: int = DEFAULT_WINDOW_STRIDE_SECONDS,
) -> tuple[ObservationDescriptor, ...]:
    """Generate fully observed causal windows with labels from future AF onset annotations."""
    if record_duration_seconds <= 0:
        raise ValueError("record_duration_seconds must be positive.")
    if stride_seconds <= 0:
        raise ValueError("stride_seconds must be positive.")

    latest_start = (
        record_duration_seconds - OBSERVATION_SECONDS - PREDICTION_HORIZON_SECONDS
    )
    if latest_start < 0:
        return ()

    af_onsets = tuple(
        interval.start_seconds for interval in rhythm_intervals if interval.is_af
    )
    descriptors: list[ObservationDescriptor] = []
    start_seconds = 0.0
    while start_seconds <= latest_start + 1e-9:
        observation_end = start_seconds + OBSERVATION_SECONDS
        prediction_end = observation_end + PREDICTION_HORIZON_SECONDS

        # This is stricter than merely excluding ongoing AF: every observation sample
        # must be covered by explicit normal/sinus rhythm annotations.
        if observation_is_explicit_sinus(
            start_seconds,
            observation_end,
            rhythm_intervals,
        ):
            label = int(
                any(
                    observation_end <= onset < prediction_end
                    for onset in af_onsets
                )
            )
            descriptors.append(
                ObservationDescriptor(
                    database=database,
                    record_id=record_id,
                    patient_id=patient_id,
                    observation_start_seconds=start_seconds,
                    observation_end_seconds=observation_end,
                    prediction_start_seconds=observation_end,
                    prediction_end_seconds=prediction_end,
                    label=label,
                )
            )
        start_seconds += stride_seconds

    return tuple(descriptors)


def select_smoke_descriptor(
    descriptors: Sequence[ObservationDescriptor],
    *,
    prefer_positive: bool = True,
) -> ObservationDescriptor:
    """Pick one deterministic descriptor for the real-record smoke test."""
    if not descriptors:
        raise ValueError("No valid sinus-rhythm observation windows were generated.")
    if prefer_positive:
        for descriptor in descriptors:
            if descriptor.label == 1:
                return descriptor
    return descriptors[0]


def segment_observation(filtered_observation: np.ndarray) -> np.ndarray:
    """Convert one 10-minute model-rate signal to [20, 1, 7500]."""
    values = np.asarray(filtered_observation, dtype=np.float64)
    expected = SEGMENTS_PER_OBSERVATION * SAMPLES_PER_SEGMENT
    if values.ndim != 1 or values.size != expected:
        raise ValueError(f"Expected {expected} samples, got {values.shape}.")
    if not np.all(np.isfinite(values)):
        raise ValueError("filtered_observation contains NaN or Inf.")

    segments = values.reshape(SEGMENTS_PER_OBSERVATION, SAMPLES_PER_SEGMENT)
    ecg = segments[:, np.newaxis, :]
    expected_shape = (
        SEGMENTS_PER_OBSERVATION,
        1,
        MODEL_RATE_HZ * SEGMENT_SECONDS,
    )
    if ecg.shape != expected_shape:
        raise RuntimeError(f"Unexpected ECG observation shape: {ecg.shape}.")
    return ecg

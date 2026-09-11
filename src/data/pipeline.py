"""AFDB-first one-record preprocessing pipeline for a real smoke test."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from data.afdb import (
    extract_af_intervals,
    load_afdb_metadata,
    load_afdb_rhythm_intervals,
    load_afdb_signal_segment,
)
from data.contracts import (
    HRV_FEATURE_COUNT,
    MODEL_RATE_HZ,
    SAMPLES_PER_SEGMENT,
    SEGMENTS_PER_OBSERVATION,
)
from data.hrv import compute_hrv_sequence
from data.preprocessing import preprocess_ecg
from data.types import ObservationWindow
from data.windows import (
    DEFAULT_WINDOW_STRIDE_SECONDS,
    generate_observation_descriptors,
    segment_observation,
    select_smoke_descriptor,
)


def build_one_afdb_observation(
    raw_dir: Path,
    record_id: str,
    *,
    lead_index: int = 0,
    stride_seconds: int = DEFAULT_WINDOW_STRIDE_SECONDS,
    prefer_positive: bool = True,
) -> tuple[ObservationWindow, int]:
    """Build exactly one model-ready AFDB observation without preprocessing the full database."""
    metadata = load_afdb_metadata(raw_dir, record_id)
    rhythm_intervals = load_afdb_rhythm_intervals(raw_dir, record_id, metadata)
    af_intervals = extract_af_intervals(rhythm_intervals)

    descriptors = generate_observation_descriptors(
        database=metadata.database,
        record_id=metadata.record_id,
        patient_id=metadata.patient_id,
        record_duration_seconds=metadata.duration_seconds,
        rhythm_intervals=rhythm_intervals,
        stride_seconds=stride_seconds,
    )
    descriptor = select_smoke_descriptor(descriptors, prefer_positive=prefer_positive)

    raw_signal, loaded_fs, lead_name = load_afdb_signal_segment(
        raw_dir,
        record_id,
        start_seconds=descriptor.observation_start_seconds,
        end_seconds=descriptor.observation_end_seconds,
        lead_index=lead_index,
        metadata=metadata,
    )
    if not np.isclose(loaded_fs, metadata.original_fs_hz):
        raise RuntimeError(
            f"Loaded fs {loaded_fs} does not match header fs {metadata.original_fs_hz}."
        )

    filtered = preprocess_ecg(
        raw_signal,
        original_fs_hz=metadata.original_fs_hz,
        target_fs_hz=MODEL_RATE_HZ,
        low_hz=0.5,
        high_hz=40.0,
    )
    expected_samples = SEGMENTS_PER_OBSERVATION * SAMPLES_PER_SEGMENT
    if filtered.size != expected_samples:
        raise RuntimeError(
            f"Expected {expected_samples} model-rate samples, got {filtered.size}."
        )

    ecg = segment_observation(filtered)
    hrv, valid_mask, reasons = compute_hrv_sequence(filtered, fs_hz=MODEL_RATE_HZ)
    if hrv.shape != (SEGMENTS_PER_OBSERVATION, HRV_FEATURE_COUNT):
        raise RuntimeError(f"Unexpected HRV shape: {hrv.shape}.")

    window = ObservationWindow(
        ecg=ecg.astype(np.float32, copy=False),
        hrv=hrv.astype(np.float32, copy=False),
        hrv_valid_mask=valid_mask,
        hrv_missing_reasons=reasons,
        label=descriptor.label,
        database=descriptor.database,
        record_id=descriptor.record_id,
        patient_id=descriptor.patient_id,
        observation_start_seconds=descriptor.observation_start_seconds,
        observation_end_seconds=descriptor.observation_end_seconds,
        prediction_start_seconds=descriptor.prediction_start_seconds,
        prediction_end_seconds=descriptor.prediction_end_seconds,
        original_fs_hz=metadata.original_fs_hz,
        model_fs_hz=MODEL_RATE_HZ,
        lead_index=lead_index,
        lead_name=lead_name,
    )
    return window, len(af_intervals)


def save_observation_npz(window: ObservationWindow, output_path: Path) -> None:
    """Save the smoke-test observation locally; the artifacts directory is git-ignored."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        ecg=window.ecg,
        hrv=window.hrv,
        hrv_valid_mask=window.hrv_valid_mask,
        label=np.asarray(window.label, dtype=np.int64),
        metadata_json=np.asarray(window.metadata_json()),
    )

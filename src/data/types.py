"""Typed containers for AFDB rhythm and observation data."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any

import numpy as np


@dataclass(frozen=True)
class RhythmInterval:
    """A rhythm interval expressed in seconds on the original record timeline."""

    start_seconds: float
    end_seconds: float
    rhythm: str

    def __post_init__(self) -> None:
        if self.start_seconds < 0:
            raise ValueError("Rhythm interval start must be non-negative.")
        if self.end_seconds <= self.start_seconds:
            raise ValueError("Rhythm interval end must be greater than start.")
        if not self.rhythm:
            raise ValueError("Rhythm label must be non-empty.")

    @property
    def is_af(self) -> bool:
        return self.rhythm.upper() == "AFIB"


@dataclass(frozen=True)
class AFDBRecordMetadata:
    """Metadata required to preserve record and subject identity."""

    database: str
    record_id: str
    patient_id: str
    original_fs_hz: float
    signal_length_samples: int
    duration_seconds: float
    signal_names: tuple[str, ...]
    units: tuple[str, ...]


@dataclass(frozen=True)
class ObservationDescriptor:
    """Causal timing and label metadata for one candidate observation."""

    database: str
    record_id: str
    patient_id: str
    observation_start_seconds: float
    observation_end_seconds: float
    prediction_start_seconds: float
    prediction_end_seconds: float
    label: int

    def __post_init__(self) -> None:
        if self.label not in (0, 1):
            raise ValueError("Observation label must be binary.")
        if self.observation_end_seconds <= self.observation_start_seconds:
            raise ValueError("Observation interval must have positive duration.")
        if self.prediction_start_seconds != self.observation_end_seconds:
            raise ValueError("Prediction horizon must start exactly at observation end.")
        if self.prediction_end_seconds <= self.prediction_start_seconds:
            raise ValueError("Prediction horizon must have positive duration.")


@dataclass(frozen=True)
class ObservationWindow:
    """One model-ready observation plus explicit HRV missingness and metadata."""

    ecg: np.ndarray
    hrv: np.ndarray
    hrv_valid_mask: np.ndarray
    hrv_missing_reasons: tuple[tuple[str | None, ...], ...]
    label: int
    database: str
    record_id: str
    patient_id: str
    observation_start_seconds: float
    observation_end_seconds: float
    prediction_start_seconds: float
    prediction_end_seconds: float
    original_fs_hz: float
    model_fs_hz: int
    lead_index: int
    lead_name: str

    def metadata_dict(self) -> dict[str, Any]:
        """Return JSON-serializable metadata without duplicating signal arrays."""
        return {
            "database": self.database,
            "record_id": self.record_id,
            "patient_id": self.patient_id,
            "observation_start_seconds": self.observation_start_seconds,
            "observation_end_seconds": self.observation_end_seconds,
            "prediction_start_seconds": self.prediction_start_seconds,
            "prediction_end_seconds": self.prediction_end_seconds,
            "label": self.label,
            "original_fs_hz": self.original_fs_hz,
            "model_fs_hz": self.model_fs_hz,
            "lead_index": self.lead_index,
            "lead_name": self.lead_name,
            "hrv_valid_mask": self.hrv_valid_mask.astype(bool).tolist(),
            "hrv_missing_reasons": [list(row) for row in self.hrv_missing_reasons],
        }

    def metadata_json(self) -> str:
        return json.dumps(self.metadata_dict(), sort_keys=True)

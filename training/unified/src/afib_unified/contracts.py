"""Immutable tensor and feature contracts."""

from dataclasses import dataclass


HRV_FEATURE_NAMES = (
    "rmssd_ms",
    "sdnn_ms",
    "lf_hf_ratio",
    "mean_rr_ms",
    "pnn50_percent",
    "sample_entropy",
)

LEAD_BINS = ("00_04", "04_08", "08_12", "12_16", "16_20")


@dataclass(frozen=True)
class PipelineContract:
    target_fs_hz: int = 250
    segment_seconds: int = 30
    segments_per_window: int = 20

    @property
    def segment_samples(self) -> int:
        return self.target_fs_hz * self.segment_seconds


DEFAULT_CONTRACT = PipelineContract()

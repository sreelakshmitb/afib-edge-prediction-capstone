"""Immutable project-level signal and tensor contracts."""

from dataclasses import dataclass


HRV_FEATURE_NAMES: tuple[str, ...] = (
    "rmssd_ms",
    "sdnn_ms",
    "lf_hf_ratio",
    "mean_rr_ms",
    "pnn50_percent",
    "sample_entropy",
)


@dataclass(frozen=True)
class PipelineContract:
    target_fs_hz: int = 250
    segment_seconds: int = 30
    segments_per_window: int = 20
    prediction_horizon_minutes: int = 20
    overlap_fraction: float = 0.20

    @property
    def segment_samples(self) -> int:
        return self.target_fs_hz * self.segment_seconds

    @property
    def observation_seconds(self) -> int:
        return self.segment_seconds * self.segments_per_window

    @property
    def observation_samples(self) -> int:
        return self.segment_samples * self.segments_per_window

    @property
    def prediction_horizon_seconds(self) -> int:
        return self.prediction_horizon_minutes * 60

    @property
    def stride_samples(self) -> int:
        return round(self.observation_samples * (1.0 - self.overlap_fraction))

    def validate(self) -> None:
        if self.target_fs_hz != 250:
            raise ValueError("The fixed project sampling rate is 250 Hz")
        if self.segment_seconds != 30 or self.segments_per_window != 20:
            raise ValueError("The fixed observation is 20 contiguous 30-second segments")
        if not 0.0 <= self.overlap_fraction < 1.0:
            raise ValueError("overlap_fraction must be in [0, 1)")


DEFAULT_CONTRACT = PipelineContract()
DEFAULT_CONTRACT.validate()


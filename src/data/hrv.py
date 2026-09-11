"""R-peak/RR extraction and six-feature HRV computation with explicit missingness."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.signal import detrend, welch

from data.contracts import HRV_FEATURE_COUNT, HRV_FEATURE_ORDER, MODEL_RATE_HZ
from data.contracts import SAMPLES_PER_SEGMENT, SEGMENTS_PER_OBSERVATION


@dataclass(frozen=True)
class HRVResult:
    """Six HRV values plus validity and a reason for every missing feature."""

    values: np.ndarray
    valid_mask: np.ndarray
    reasons: tuple[str | None, ...]


def _all_missing(reason: str) -> HRVResult:
    return HRVResult(
        values=np.full(HRV_FEATURE_COUNT, np.nan, dtype=np.float64),
        valid_mask=np.zeros(HRV_FEATURE_COUNT, dtype=bool),
        reasons=tuple(reason for _ in range(HRV_FEATURE_COUNT)),
    )


def detect_r_peaks(signal: np.ndarray, *, fs_hz: float = MODEL_RATE_HZ) -> np.ndarray:
    """Detect QRS/R-peak locations using WFDB XQRS on preprocessed ECG."""
    from wfdb import processing

    values = np.asarray(signal, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("signal must be a non-empty 1-D array.")
    if not np.all(np.isfinite(values)):
        raise ValueError("signal contains NaN or Inf.")
    if fs_hz <= 0:
        raise ValueError("fs_hz must be positive.")

    peaks = processing.xqrs_detect(sig=values, fs=fs_hz, verbose=False)
    peaks = np.asarray(peaks, dtype=np.int64)
    if peaks.ndim != 1:
        raise RuntimeError("XQRS returned an invalid peak array.")
    return peaks


def rr_intervals_ms(r_peaks: np.ndarray, *, fs_hz: float = MODEL_RATE_HZ) -> np.ndarray:
    """Convert increasing R-peak sample indices to RR intervals in milliseconds."""
    peaks = np.asarray(r_peaks, dtype=np.int64)
    if peaks.ndim != 1:
        raise ValueError("r_peaks must be one-dimensional.")
    if fs_hz <= 0:
        raise ValueError("fs_hz must be positive.")
    if peaks.size < 2:
        return np.empty(0, dtype=np.float64)
    differences = np.diff(peaks)
    if np.any(differences <= 0):
        raise ValueError("r_peaks must be strictly increasing.")
    return differences.astype(np.float64) * (1000.0 / fs_hz)


def _sample_entropy(
    values: np.ndarray,
    *,
    m: int = 2,
    tolerance_ratio: float = 0.2,
) -> float:
    data = np.asarray(values, dtype=np.float64)
    if data.ndim != 1 or data.size < 8:
        return math.nan
    sd = float(np.std(data, ddof=1))
    if not math.isfinite(sd) or sd <= 0:
        return math.nan
    tolerance = tolerance_ratio * sd

    def match_probability(template_length: int) -> float:
        template_count = data.size - template_length + 1
        if template_count < 2:
            return 0.0
        matches = 0
        total_pairs = template_count * (template_count - 1) // 2
        for first in range(template_count - 1):
            template = data[first : first + template_length]
            for second in range(first + 1, template_count):
                candidate = data[second : second + template_length]
                if np.max(np.abs(template - candidate)) <= tolerance:
                    matches += 1
        return matches / total_pairs if total_pairs else 0.0

    probability_m = match_probability(m)
    probability_m1 = match_probability(m + 1)
    if probability_m <= 0 or probability_m1 <= 0:
        return math.nan
    return float(-math.log(probability_m1 / probability_m))


def _lf_hf_ratio(rr_ms: np.ndarray) -> float:
    # REPRODUCTION CHOICE: 30-s HRV is retained because the paper uses per-subsegment
    # six-dimensional HRV, but LF/HF is marked missing when support is too weak.
    if rr_ms.size < 10:
        return math.nan
    rr_seconds = rr_ms / 1000.0
    beat_times = np.cumsum(rr_seconds)
    beat_times -= beat_times[0]
    if beat_times[-1] < 20.0:
        return math.nan

    interpolation_hz = 4.0
    grid = np.arange(0.0, beat_times[-1], 1.0 / interpolation_hz)
    if grid.size < 16:
        return math.nan
    interpolated = np.interp(grid, beat_times, rr_ms)
    interpolated = detrend(interpolated, type="constant")
    frequencies, power = welch(
        interpolated,
        fs=interpolation_hz,
        nperseg=min(128, interpolated.size),
        detrend="constant",
        scaling="density",
    )

    lf_mask = (frequencies >= 0.04) & (frequencies < 0.15)
    hf_mask = (frequencies >= 0.15) & (frequencies <= 0.40)
    if np.count_nonzero(lf_mask) < 1 or np.count_nonzero(hf_mask) < 1:
        return math.nan
    lf_power = float(np.trapz(power[lf_mask], frequencies[lf_mask]))
    hf_power = float(np.trapz(power[hf_mask], frequencies[hf_mask]))
    if not math.isfinite(lf_power) or not math.isfinite(hf_power) or hf_power <= 1e-12:
        return math.nan
    ratio = lf_power / hf_power
    return ratio if math.isfinite(ratio) and ratio >= 0 else math.nan


def compute_hrv_features(rr_ms: np.ndarray) -> HRVResult:
    """Compute HRV in fixed order: RMSSD, SDNN, LF/HF, Mean RR, pNN50, Sample Entropy."""
    rr = np.asarray(rr_ms, dtype=np.float64)
    if rr.ndim != 1:
        raise ValueError("rr_ms must be one-dimensional.")
    if rr.size == 0:
        return _all_missing("insufficient_rr")
    if not np.all(np.isfinite(rr)) or np.any(rr <= 0):
        return _all_missing("invalid_rr")

    values = np.full(HRV_FEATURE_COUNT, np.nan, dtype=np.float64)
    reasons: list[str | None] = ["insufficient_rr"] * HRV_FEATURE_COUNT

    values[3] = float(np.mean(rr))
    reasons[3] = None

    rr_diff = np.diff(rr)
    if rr_diff.size >= 1:
        values[0] = float(np.sqrt(np.mean(np.square(rr_diff))))
        reasons[0] = None
        values[1] = float(np.std(rr, ddof=1))
        reasons[1] = None
        values[4] = float(100.0 * np.mean(np.abs(rr_diff) > 50.0))
        reasons[4] = None

    lf_hf = _lf_hf_ratio(rr)
    if math.isfinite(lf_hf):
        values[2] = lf_hf
        reasons[2] = None
    else:
        reasons[2] = "unstable_lf_hf"

    sample_entropy = _sample_entropy(rr)
    if math.isfinite(sample_entropy):
        values[5] = sample_entropy
        reasons[5] = None
    else:
        reasons[5] = "sample_entropy_unavailable"

    valid_mask = np.isfinite(values)
    for index, valid in enumerate(valid_mask):
        if valid:
            reasons[index] = None
        elif reasons[index] is None:
            reasons[index] = "non_finite_result"

    return HRVResult(values=values, valid_mask=valid_mask, reasons=tuple(reasons))


def compute_hrv_sequence(
    filtered_observation: np.ndarray,
    *,
    fs_hz: int = MODEL_RATE_HZ,
) -> tuple[np.ndarray, np.ndarray, tuple[tuple[str | None, ...], ...]]:
    """Compute per-30-s HRV for a complete 20-segment observation."""
    values = np.asarray(filtered_observation, dtype=np.float64)
    expected_length = SEGMENTS_PER_OBSERVATION * SAMPLES_PER_SEGMENT
    if values.ndim != 1 or values.size != expected_length:
        raise ValueError(
            f"Expected a 1-D {expected_length}-sample observation, got {values.shape}."
        )
    if not np.all(np.isfinite(values)):
        raise ValueError("filtered_observation contains NaN or Inf.")

    try:
        peaks = detect_r_peaks(values, fs_hz=fs_hz)
    except Exception as exc:  # explicit missingness is preferable to fake HRV values
        reason = f"r_peak_detection_failed:{type(exc).__name__}"
        missing = _all_missing(reason)
        hrv = np.tile(missing.values, (SEGMENTS_PER_OBSERVATION, 1))
        mask = np.tile(missing.valid_mask, (SEGMENTS_PER_OBSERVATION, 1))
        reasons = tuple(missing.reasons for _ in range(SEGMENTS_PER_OBSERVATION))
        return hrv, mask, reasons

    rows: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    reasons: list[tuple[str | None, ...]] = []
    for segment_index in range(SEGMENTS_PER_OBSERVATION):
        start = segment_index * SAMPLES_PER_SEGMENT
        end = start + SAMPLES_PER_SEGMENT
        segment_peaks = peaks[(peaks >= start) & (peaks < end)] - start
        rr = rr_intervals_ms(segment_peaks, fs_hz=fs_hz)
        result = compute_hrv_features(rr)
        rows.append(result.values)
        masks.append(result.valid_mask)
        reasons.append(result.reasons)

    return np.stack(rows), np.stack(masks), tuple(reasons)


assert HRV_FEATURE_ORDER == (
    "RMSSD",
    "SDNN",
    "LF/HF",
    "Mean RR",
    "pNN50",
    "Sample Entropy",
)

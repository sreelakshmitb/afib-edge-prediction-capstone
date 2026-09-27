"""Six-feature HRV extraction with a fixed, testable feature order."""

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import welch

from .contracts import HRV_FEATURE_NAMES


@dataclass(frozen=True)
class HRVResult:
    values: np.ndarray
    valid_rr_count: int
    rejected_rr_count: int

    def __post_init__(self) -> None:
        if self.values.shape != (len(HRV_FEATURE_NAMES),):
            raise ValueError("HRV values do not match the fixed six-feature contract")


def _sample_entropy(values: np.ndarray, embedding_dim: int = 2, tolerance: float | None = None) -> float:
    """Compute finite-sample-capped SampEn(m, r) without self-matches.

    The ordinary count-ratio estimator is used whenever both match counts are
    non-zero.  For short sequences, a zero ``m + 1`` match count otherwise
    produces positive infinity; it is replaced by a one-match upper bound.  If
    even the ``m`` count is zero, the result is capped using all possible
    ``m``-template pairs.  A perfectly constant sequence has zero entropy.

    This keeps finite-sample estimator failures distinct from genuinely
    missing HRV caused by too few usable RR intervals.
    """
    x = np.asarray(values, dtype=np.float64)
    if len(x) <= embedding_dim + 1:
        return np.nan
    r = tolerance if tolerance is not None else 0.2 * np.std(x, ddof=1)
    if not np.isfinite(r):
        return np.nan
    if r <= 0:
        return 0.0 if np.all(x == x[0]) else np.nan

    def matches(length: int) -> int:
        templates = np.array([x[i : i + length] for i in range(len(x) - length + 1)])
        count = 0
        for i in range(len(templates) - 1):
            distance = np.max(np.abs(templates[i + 1 :] - templates[i]), axis=1)
            count += int(np.sum(distance <= r))
        return count

    b = matches(embedding_dim)
    a = matches(embedding_dim + 1)
    if b == 0:
        template_count = len(x) - embedding_dim + 1
        possible_pairs = template_count * (template_count - 1) // 2
        return float(np.log(max(possible_pairs, 1)))
    if a == 0:
        return float(np.log(b + 1.0))
    return float(-np.log(a / b))


def _lf_hf_ratio(rr_ms: np.ndarray) -> float:
    """Estimate LF/HF from an interpolated RR tachogram.

    Thirty-second segments provide weak LF resolution, so invalid estimates are
    returned as NaN and must be imputed from training patients only.
    """
    if len(rr_ms) < 8:
        return np.nan
    beat_times = np.cumsum(rr_ms) / 1000.0
    beat_times -= beat_times[0]
    if beat_times[-1] < 20.0:
        return np.nan
    grid_hz = 4.0
    grid = np.arange(0.0, beat_times[-1], 1.0 / grid_hz)
    if len(grid) < 16:
        return np.nan
    tachogram = interp1d(beat_times, rr_ms, kind="linear", bounds_error=False,
                         fill_value="extrapolate")(grid)
    tachogram -= np.mean(tachogram)
    frequencies, power = welch(tachogram, fs=grid_hz, nperseg=min(256, len(tachogram)))
    lf = np.trapezoid(power[(frequencies >= 0.04) & (frequencies < 0.15)],
                      frequencies[(frequencies >= 0.04) & (frequencies < 0.15)])
    hf = np.trapezoid(power[(frequencies >= 0.15) & (frequencies <= 0.40)],
                      frequencies[(frequencies >= 0.15) & (frequencies <= 0.40)])
    return float(lf / hf) if hf > 0 and np.isfinite(lf) and np.isfinite(hf) else np.nan


def compute_hrv_features(
    rpeak_samples: np.ndarray,
    fs_hz: float,
    min_rr_ms: float = 300.0,
    max_rr_ms: float = 2000.0,
) -> HRVResult:
    """Return HRV in the project order.

    Invalid or unsupported features remain NaN. Replacing them with zeros before
    splitting would encode missingness incorrectly; fold-specific preprocessing
    will learn imputation values from training patients only.
    """
    peaks = np.asarray(rpeak_samples, dtype=np.int64)
    if peaks.ndim != 1 or fs_hz <= 0:
        raise ValueError("rpeak_samples must be one-dimensional and fs_hz positive")
    if len(peaks) < 3:
        return HRVResult(np.full(6, np.nan, dtype=np.float32), 0, 0)
    if np.any(np.diff(peaks) <= 0):
        raise ValueError("R-peak samples must be strictly increasing")

    all_rr = np.diff(peaks) * 1000.0 / fs_hz
    valid_mask = (all_rr >= min_rr_ms) & (all_rr <= max_rr_ms) & np.isfinite(all_rr)
    rr = all_rr[valid_mask]
    rejected = int(len(all_rr) - len(rr))
    if len(rr) < 3:
        return HRVResult(np.full(6, np.nan, dtype=np.float32), len(rr), rejected)

    differences = np.diff(rr)
    values = np.array(
        [
            np.sqrt(np.mean(differences**2)) if len(differences) else np.nan,
            np.std(rr, ddof=1) if len(rr) > 1 else np.nan,
            _lf_hf_ratio(rr),
            np.mean(rr),
            100.0 * np.mean(np.abs(differences) > 50.0) if len(differences) else np.nan,
            _sample_entropy(rr),
        ],
        dtype=np.float32,
    )
    return HRVResult(values, len(rr), rejected)

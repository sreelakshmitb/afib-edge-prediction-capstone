"""Deterministic ECG filtering and resampling."""

from fractions import Fraction

import numpy as np
from scipy.signal import butter, resample_poly, sosfiltfilt


def bandpass_filter(
    signal: np.ndarray,
    fs_hz: float,
    low_hz: float = 0.5,
    high_hz: float = 40.0,
    order: int = 4,
) -> np.ndarray:
    """Apply a zero-phase Butterworth band-pass in second-order-section form."""
    x = np.asarray(signal, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError("single-lead ECG must be one-dimensional")
    if not np.isfinite(x).all():
        raise ValueError("ECG contains NaN or infinite samples")
    if not 0.0 < low_hz < high_hz < fs_hz / 2.0:
        raise ValueError("band-pass cutoffs must satisfy 0 < low < high < Nyquist")
    sos = butter(order, (low_hz, high_hz), btype="bandpass", fs=fs_hz, output="sos")
    return sosfiltfilt(sos, x).astype(np.float32, copy=False)


def resample_ecg(signal: np.ndarray, fs_in_hz: float, fs_out_hz: int = 250) -> np.ndarray:
    """Resample with an anti-aliasing polyphase filter and deterministic length."""
    x = np.asarray(signal)
    if x.ndim != 1:
        raise ValueError("single-lead ECG must be one-dimensional")
    if fs_in_hz <= 0 or fs_out_hz <= 0:
        raise ValueError("sampling frequencies must be positive")
    if float(fs_in_hz) == float(fs_out_hz):
        return x.astype(np.float32, copy=True)

    ratio = Fraction(str(fs_out_hz)) / Fraction(str(fs_in_hz))
    ratio = ratio.limit_denominator(10_000)
    y = resample_poly(x, ratio.numerator, ratio.denominator)
    expected = round(len(x) * fs_out_hz / fs_in_hz)
    if len(y) > expected:
        y = y[:expected]
    elif len(y) < expected:
        y = np.pad(y, (0, expected - len(y)), mode="edge")
    return y.astype(np.float32, copy=False)


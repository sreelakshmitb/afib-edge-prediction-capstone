"""Deterministic ECG resampling and band-pass preprocessing."""

from __future__ import annotations

from fractions import Fraction

import numpy as np
from scipy.signal import butter, resample_poly, sosfiltfilt

from data.contracts import MODEL_RATE_HZ


def resample_signal(
    signal: np.ndarray,
    *,
    original_fs_hz: float,
    target_fs_hz: float = MODEL_RATE_HZ,
) -> np.ndarray:
    """Resample a 1-D ECG signal using polyphase filtering and deterministic length handling."""
    values = np.asarray(signal, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("signal must be one-dimensional.")
    if values.size == 0:
        raise ValueError("signal must not be empty.")
    if not np.all(np.isfinite(values)):
        raise ValueError("signal contains NaN or Inf.")
    if original_fs_hz <= 0 or target_fs_hz <= 0:
        raise ValueError("Sampling rates must be positive.")
    if np.isclose(original_fs_hz, target_fs_hz):
        return values.copy()

    ratio = Fraction(target_fs_hz / original_fs_hz).limit_denominator(10_000)
    resampled = resample_poly(values, ratio.numerator, ratio.denominator)
    expected_length = int(round(values.size * target_fs_hz / original_fs_hz))

    if resampled.size > expected_length:
        resampled = resampled[:expected_length]
    elif resampled.size < expected_length:
        shortfall = expected_length - resampled.size
        if shortfall > 1:
            raise RuntimeError(
                f"Unexpected resampling length: got {resampled.size}, expected {expected_length}."
            )
        resampled = np.pad(resampled, (0, shortfall), mode="edge")

    return np.asarray(resampled, dtype=np.float64)


def bandpass_filter(
    signal: np.ndarray,
    *,
    fs_hz: float = MODEL_RATE_HZ,
    low_hz: float = 0.5,
    high_hz: float = 40.0,
    order: int = 4,
) -> np.ndarray:
    """Apply a zero-phase Butterworth band-pass filter."""
    values = np.asarray(signal, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("signal must be one-dimensional.")
    if values.size == 0:
        raise ValueError("signal must not be empty.")
    if not np.all(np.isfinite(values)):
        raise ValueError("signal contains NaN or Inf.")
    if fs_hz <= 0:
        raise ValueError("fs_hz must be positive.")
    if not 0 < low_hz < high_hz < fs_hz / 2:
        raise ValueError("Band-pass cutoffs must satisfy 0 < low < high < Nyquist.")
    if order < 1:
        raise ValueError("Filter order must be positive.")

    sos = butter(order, (low_hz, high_hz), btype="bandpass", fs=fs_hz, output="sos")
    filtered = sosfiltfilt(sos, values)
    if filtered.shape != values.shape or not np.all(np.isfinite(filtered)):
        raise RuntimeError("Band-pass filtering produced invalid output.")
    return np.asarray(filtered, dtype=np.float64)


def preprocess_ecg(
    signal: np.ndarray,
    *,
    original_fs_hz: float,
    target_fs_hz: int = MODEL_RATE_HZ,
    low_hz: float = 0.5,
    high_hz: float = 40.0,
) -> np.ndarray:
    """Paper-aligned preprocessing order: resample to model rate, then band-pass filter."""
    resampled = resample_signal(
        signal,
        original_fs_hz=original_fs_hz,
        target_fs_hz=target_fs_hz,
    )
    return bandpass_filter(
        resampled,
        fs_hz=target_fs_hz,
        low_hz=low_hz,
        high_hz=high_hz,
    )

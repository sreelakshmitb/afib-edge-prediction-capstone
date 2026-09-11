import numpy as np

from data.preprocessing import bandpass_filter, resample_signal
from data.windows import segment_observation


def test_resampling_length() -> None:
    original_fs = 128
    target_fs = 250
    duration_seconds = 4
    t = np.arange(original_fs * duration_seconds) / original_fs
    signal = np.sin(2 * np.pi * 5.0 * t)

    resampled = resample_signal(
        signal,
        original_fs_hz=original_fs,
        target_fs_hz=target_fs,
    )

    assert resampled.shape == (target_fs * duration_seconds,)
    assert np.all(np.isfinite(resampled))


def test_filter_output_shape_and_finite_values() -> None:
    fs = 250
    t = np.arange(fs * 10) / fs
    signal = np.sin(2 * np.pi * 5.0 * t) + 0.2 * np.sin(2 * np.pi * 60.0 * t)

    filtered = bandpass_filter(signal, fs_hz=fs)

    assert filtered.shape == signal.shape
    assert np.all(np.isfinite(filtered))


def test_observation_segment_shape_and_length() -> None:
    signal = np.arange(20 * 7500, dtype=np.float64)
    ecg = segment_observation(signal)

    assert ecg.shape == (20, 1, 7500)
    assert all(segment.shape == (1, 7500) for segment in ecg)

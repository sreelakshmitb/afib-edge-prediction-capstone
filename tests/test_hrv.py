import numpy as np

from data.contracts import HRV_FEATURE_ORDER
from data.hrv import compute_hrv_features


def test_hrv_feature_order_is_exact() -> None:
    assert HRV_FEATURE_ORDER == (
        "RMSSD",
        "SDNN",
        "LF/HF",
        "Mean RR",
        "pNN50",
        "Sample Entropy",
    )


def test_hrv_missingness_is_explicit_for_insufficient_rr() -> None:
    result = compute_hrv_features(np.array([800.0]))

    assert result.values.shape == (6,)
    assert result.valid_mask.shape == (6,)
    assert np.isnan(result.values[0])
    assert np.isnan(result.values[1])
    assert np.isnan(result.values[2])
    assert result.values[3] == 800.0
    assert np.isnan(result.values[4])
    assert np.isnan(result.values[5])
    assert result.reasons[2] == "unstable_lf_hf"
    assert result.reasons[5] == "sample_entropy_unavailable"

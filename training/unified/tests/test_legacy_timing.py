import numpy as np

from afib_unified.legacy_timing import _negative_policy, normalize_rhythm


def test_legacy_timing_policy_matches_iridia_contract():
    assert normalize_rhythm("(AFIB") == "AFIB"
    assert normalize_rhythm("(N\x00") == "N"
    assert normalize_rhythm("") is None
    assert _negative_policy(35.0, np.nan) == ("near_20_60", True, "")
    assert _negative_policy(120.0, np.nan) == ("medium_60_360", True, "")
    assert _negative_policy(np.nan, np.nan) == ("far_360_plus_or_none", True, "")
    stratum, eligible, reason = _negative_policy(500.0, 12.0)
    assert stratum == "post_af_recovery_excluded"
    assert eligible is False
    assert "within_30m" in reason

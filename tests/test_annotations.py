from data.afdb import extract_af_intervals, extract_rhythm_intervals


def test_annotation_timing_uses_original_sampling_rate() -> None:
    intervals = extract_rhythm_intervals(
        [0, 25_000, 40_000],
        ["(N", "(AFIB", "(N"],
        original_fs_hz=250.0,
        record_duration_seconds=200.0,
    )
    af = extract_af_intervals(intervals)

    assert len(af) == 1
    assert af[0].start_seconds == 100.0
    assert af[0].end_seconds == 160.0
    assert af[0].rhythm == "AFIB"

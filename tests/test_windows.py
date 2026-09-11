from data.types import RhythmInterval
from data.windows import generate_observation_descriptors


def test_label_causality_positive_only_for_future_af_onset() -> None:
    rhythms = (
        RhythmInterval(0.0, 600.0, "N"),
        RhythmInterval(600.0, 900.0, "AFIB"),
        RhythmInterval(900.0, 3000.0, "N"),
    )
    descriptors = generate_observation_descriptors(
        database="afdb",
        record_id="04015",
        patient_id="04015",
        record_duration_seconds=3000.0,
        rhythm_intervals=rhythms,
        stride_seconds=600,
    )

    first = descriptors[0]
    assert first.observation_start_seconds == 0.0
    assert first.observation_end_seconds == 600.0
    assert first.prediction_start_seconds == 600.0
    assert first.prediction_end_seconds == 1800.0
    assert first.label == 1


def test_observation_overlapping_ongoing_af_is_excluded() -> None:
    rhythms = (
        RhythmInterval(0.0, 300.0, "N"),
        RhythmInterval(300.0, 700.0, "AFIB"),
        RhythmInterval(700.0, 3000.0, "N"),
    )
    descriptors = generate_observation_descriptors(
        database="afdb",
        record_id="04015",
        patient_id="04015",
        record_duration_seconds=3000.0,
        rhythm_intervals=rhythms,
        stride_seconds=600,
    )

    assert all(item.observation_start_seconds != 0.0 for item in descriptors)


def test_patient_and_record_identity_are_preserved() -> None:
    rhythms = (RhythmInterval(0.0, 3000.0, "N"),)
    descriptors = generate_observation_descriptors(
        database="afdb",
        record_id="record-x",
        patient_id="patient-x",
        record_duration_seconds=3000.0,
        rhythm_intervals=rhythms,
        stride_seconds=600,
    )

    assert descriptors
    assert descriptors[0].database == "afdb"
    assert descriptors[0].record_id == "record-x"
    assert descriptors[0].patient_id == "patient-x"


def test_af_onset_inside_observation_is_not_mislabeled_positive() -> None:
    rhythms = (
        RhythmInterval(0.0, 500.0, "N"),
        RhythmInterval(500.0, 700.0, "AFIB"),
        RhythmInterval(700.0, 3000.0, "N"),
    )
    descriptors = generate_observation_descriptors(
        database="afdb",
        record_id="04015",
        patient_id="04015",
        record_duration_seconds=3000.0,
        rhythm_intervals=rhythms,
        stride_seconds=600,
    )

    assert all(item.observation_start_seconds != 0.0 for item in descriptors)

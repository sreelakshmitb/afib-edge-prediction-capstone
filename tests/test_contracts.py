from data.contracts import (
    HRV_FEATURE_COUNT,
    HRV_FEATURE_ORDER,
    MAX_MODEL_PARAMETERS_EXCLUSIVE,
    MODEL_RATE_HZ,
    OBSERVATION_MINUTES,
    OUTER_GROUP_KFOLD_SPLITS,
    PREDICTION_HORIZON_MINUTES,
    SAMPLES_PER_SEGMENT,
    SEGMENT_SECONDS,
    SEGMENTS_PER_OBSERVATION,
    validate_contracts,
)


def test_signal_contract() -> None:
    assert MODEL_RATE_HZ == 250
    assert SEGMENT_SECONDS == 30
    assert SAMPLES_PER_SEGMENT == 7_500
    assert SEGMENTS_PER_OBSERVATION == 20
    assert OBSERVATION_MINUTES == 10
    assert PREDICTION_HORIZON_MINUTES == 20


def test_hrv_contract() -> None:
    assert HRV_FEATURE_COUNT == 6
    assert HRV_FEATURE_ORDER == (
        "RMSSD",
        "SDNN",
        "LF/HF",
        "Mean RR",
        "pNN50",
        "Sample Entropy",
    )


def test_evaluation_and_model_constraints() -> None:
    assert OUTER_GROUP_KFOLD_SPLITS == 5
    assert MAX_MODEL_PARAMETERS_EXCLUSIVE == 250_000
    validate_contracts()

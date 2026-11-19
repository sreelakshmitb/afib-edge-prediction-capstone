import numpy as np

from validation_pilot import classification, select_threshold


def test_threshold_rule_respects_specificity_floor():
    labels = np.array([1, 1] + [0] * 10, dtype=np.uint8)
    probabilities = np.array([0.9, 0.7, 0.8, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.02, 0.01])
    selected = select_threshold(labels, probabilities)
    assert selected["specificity"] >= 0.90
    assert selected["recall"] == 1.0
    check = classification(labels, probabilities, selected["threshold"])
    assert check == selected

import numpy as np
import pandas as pd

from afib_unified.validation_metrics import select_threshold, summarize_predictions


def test_diagnostics_include_event_and_patient_metrics():
    rows = []
    for source in ("iridia", "legacy"):
        for patient in range(2):
            for label, probability in ((1, 0.9 - patient * 0.1), (0, 0.1 + patient * 0.1)):
                rows.append({
                    "window_id": f"{source}:{patient}:{label}",
                    "subject_id": f"{source}:p{patient}",
                    "source_family": source,
                    "label": label,
                    "event_id": f"{source}:e{patient}" if label else "",
                    "lead_bin": "00_04" if label else "",
                    "negative_stratum": "near_20_60" if source == "iridia" else "legacy_unstratified",
                    "probability": probability,
                })
    summary = summarize_predictions(pd.DataFrame(rows))
    assert summary["ranking"]["pr_auc"] == 1.0
    assert summary["global_operating_point"]["specificity"] >= 0.90
    for source in ("iridia", "legacy"):
        assert summary["by_source"][source]["event_sensitivity_at_source_threshold"]["events"] == 2
        assert summary["by_source"][source]["patient_macro_at_source_threshold"]["positive_patients"] == 2


def test_threshold_maximizes_recall_under_specificity_floor():
    labels = np.array([1, 1] + [0] * 10, dtype=np.uint8)
    probabilities = np.array([0.9, 0.7, 0.8, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.02, 0.01])
    selected = select_threshold(labels, probabilities)
    assert selected["specificity"] >= 0.90
    assert selected["recall"] == 1.0

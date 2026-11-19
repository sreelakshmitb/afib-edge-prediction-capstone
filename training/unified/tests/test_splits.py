import numpy as np
import pandas as pd

from afib_unified.manifest import freeze_splits


def test_patient_wise_frozen_splits_have_no_overlap():
    rows = []
    for source in ("legacy", "iridia"):
        for patient in range(50):
            for window in range(4):
                rows.append({
                    "subject_id": f"{source}:p{patient}",
                    "source_family": source,
                    "label": int(window == 0 and patient % 2 == 0),
                })
    frame, summary = freeze_splits(pd.DataFrame(rows), seed=7)
    assert summary["all_gates_pass"]
    assert frame.groupby("subject_id").outer_test_fold.nunique().eq(1).all()
    assert set(frame.outer_test_fold.unique()) == set(range(5))
    for fold in summary["folds"]:
        train = set(fold["train_subjects"])
        validation = set(fold["validation_subjects"])
        test = set(fold["test_subjects"])
        assert not train & validation
        assert not (train | validation) & test

"""Patient-wise outer cross-validation manifests and invariants."""

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from sklearn.model_selection import GroupKFold

from .windowing import WindowRecord


@dataclass(frozen=True)
class OuterFold:
    fold: int
    train_indices: tuple[int, ...]
    test_indices: tuple[int, ...]
    train_subjects: tuple[str, ...]
    test_subjects: tuple[str, ...]


def assert_patient_disjoint(train_subjects: Sequence[str], test_subjects: Sequence[str]) -> None:
    overlap = set(train_subjects) & set(test_subjects)
    if overlap:
        raise AssertionError(f"patient leakage detected: {sorted(overlap)}")


def make_outer_group_folds(
    windows: Sequence[WindowRecord], n_splits: int = 5
) -> tuple[OuterFold, ...]:
    if not windows:
        raise ValueError("cannot split an empty window manifest")
    groups = np.asarray([window.subject_id for window in windows])
    labels = np.asarray([window.label for window in windows])
    if len(np.unique(groups)) < n_splits:
        raise ValueError("number of unique patients must be at least n_splits")

    folds: list[OuterFold] = []
    splitter = GroupKFold(n_splits=n_splits)
    dummy = np.zeros(len(windows), dtype=np.uint8)
    for fold, (train_idx, test_idx) in enumerate(splitter.split(dummy, labels, groups)):
        train_subjects = tuple(sorted(set(groups[train_idx])))
        test_subjects = tuple(sorted(set(groups[test_idx])))
        assert_patient_disjoint(train_subjects, test_subjects)
        folds.append(
            OuterFold(
                fold=fold,
                train_indices=tuple(int(i) for i in train_idx),
                test_indices=tuple(int(i) for i in test_idx),
                train_subjects=train_subjects,
                test_subjects=test_subjects,
            )
        )

    test_appearances = np.zeros(len(windows), dtype=np.uint8)
    for fold in folds:
        test_appearances[list(fold.test_indices)] += 1
    if not np.all(test_appearances == 1):
        raise AssertionError("each window must appear in exactly one outer-test fold")
    return tuple(folds)


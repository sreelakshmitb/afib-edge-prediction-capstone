"""Build and freeze a cache-addressable, patient-wise unified manifest."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    mapped = series.astype(str).str.lower().map(
        {"true": True, "false": False, "1": True, "0": False}
    )
    if mapped.isna().any():
        raise ValueError("training_eligible contains non-boolean values")
    return mapped.astype(bool)


def build_cached_manifest(combined_path: Path, iridia_plan_path: Path) -> pd.DataFrame:
    combined = pd.read_csv(combined_path)
    iridia = pd.read_csv(iridia_plan_path)
    legacy = combined[combined["source_family"].astype(str).eq("legacy")].copy()
    if len(legacy) != 4862 or int(legacy["label"].sum()) != 368:
        raise ValueError("legacy manifest does not match the locked 4,862/368 cache")
    if len(iridia) != 3430 or int(iridia["label"].sum()) != 686:
        raise ValueError("IRIDIA plan does not match the locked 3,430/686 cache")

    legacy = legacy.sort_values(
        ["subject_id", "start_sample_250hz"], kind="stable"
    ).reset_index(drop=True)
    legacy["cache_family"] = "legacy"
    legacy["cache_record_id"] = legacy["subject_id"].astype(str)
    legacy["cache_local_index"] = legacy.groupby("subject_id", sort=False).cumcount()
    legacy["window_id"] = [
        f"legacy:{subject}:{int(start)}"
        for subject, start in zip(legacy.subject_id, legacy.start_sample_250hz)
    ]

    iridia = iridia.sort_values(
        ["record_id", "observation_start_seconds"], kind="stable"
    ).reset_index(drop=True)
    iridia["cache_family"] = "iridia"
    iridia["cache_record_id"] = iridia["record_id"].astype(str)
    iridia["cache_local_index"] = iridia.groupby("record_id", sort=False).cumcount()
    iridia["source_family"] = "iridia"
    iridia["dataset"] = "iridia"

    required = {
        "dataset", "source_family", "record_id", "subject_id", "label",
        "event_id", "lead_bin", "negative_stratum", "training_eligible",
        "start_sample_250hz", "window_id", "cache_family", "cache_record_id",
        "cache_local_index",
    }
    for name, frame in (("legacy", legacy), ("iridia", iridia)):
        if missing := required - set(frame.columns):
            raise ValueError(f"{name} manifest missing columns: {sorted(missing)}")
        frame["training_eligible"] = _as_bool(frame["training_eligible"])

    columns = sorted(required)
    unified = pd.concat([legacy[columns], iridia[columns]], ignore_index=True)
    unified.insert(0, "manifest_index", np.arange(len(unified), dtype=np.int64))
    if unified.window_id.duplicated().any():
        raise ValueError("duplicate unified window IDs")
    if unified.groupby("subject_id")["source_family"].nunique().gt(1).any():
        raise ValueError("cross-source subject collision")
    if len(unified) != 8292 or int(unified.label.sum()) != 1054:
        raise AssertionError("unified totals do not reconcile")
    return unified


def freeze_splits(manifest: pd.DataFrame, seed: int = 7) -> tuple[pd.DataFrame, dict]:
    frame = manifest.copy()
    y = frame.label.to_numpy(dtype=np.int8)
    groups = frame.subject_id.astype(str).to_numpy()
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    assignments = np.full(len(frame), -1, dtype=np.int8)
    split_rows = []
    all_indices = np.arange(len(frame))
    for fold, (outer_train, outer_test) in enumerate(outer.split(all_indices, y, groups)):
        assignments[outer_test] = fold
        inner = StratifiedGroupKFold(
            n_splits=5, shuffle=True, random_state=seed + 1000 + fold
        )
        relative_train, relative_validation = next(
            inner.split(outer_train, y[outer_train], groups[outer_train])
        )
        train = outer_train[relative_train]
        validation = outer_train[relative_validation]
        train_subjects = sorted(set(groups[train]))
        validation_subjects = sorted(set(groups[validation]))
        test_subjects = sorted(set(groups[outer_test]))
        if set(train_subjects) & set(validation_subjects):
            raise AssertionError("inner train/validation patient leakage")
        if (set(train_subjects) | set(validation_subjects)) & set(test_subjects):
            raise AssertionError("outer-test patient leakage")

        def describe(indices: np.ndarray) -> dict:
            subset = frame.iloc[indices]
            return {
                "windows": int(len(subset)),
                "positives": int(subset.label.sum()),
                "patients": int(subset.subject_id.nunique()),
                "source_windows": {
                    str(k): int(v) for k, v in subset.source_family.value_counts().sort_index().items()
                },
                "source_positives": {
                    str(k): int(v) for k, v in subset.groupby("source_family").label.sum().sort_index().items()
                },
            }

        split_rows.append(
            {
                "fold": fold,
                "train_subjects": train_subjects,
                "validation_subjects": validation_subjects,
                "test_subjects": test_subjects,
                "train": describe(train),
                "validation": describe(validation),
                "test": describe(outer_test),
            }
        )
    if np.any(assignments < 0):
        raise AssertionError("some windows lack an outer-test fold")
    frame["outer_test_fold"] = assignments
    if not frame.groupby("subject_id").outer_test_fold.nunique().eq(1).all():
        raise AssertionError("a patient appears in multiple outer-test folds")
    gates = {
        "five_outer_folds": len(split_rows) == 5,
        "each_patient_in_one_outer_fold": bool(
            frame.groupby("subject_id").outer_test_fold.nunique().eq(1).all()
        ),
        "every_split_has_both_sources": all(
            set(row[part]["source_windows"]) == {"iridia", "legacy"}
            for row in split_rows for part in ("train", "validation", "test")
        ),
        "every_split_has_positives": all(
            row[part]["positives"] > 0
            for row in split_rows for part in ("train", "validation", "test")
        ),
    }
    return frame, {
        "format": "afib_unified_patient_splits_v1",
        "seed": seed,
        "folds": split_rows,
        "gates": gates,
        "all_gates_pass": all(gates.values()),
    }


def build_and_save(
    combined_path: Path, iridia_plan_path: Path, output_dir: Path, seed: int = 7
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = build_cached_manifest(combined_path, iridia_plan_path)
    manifest, splits = freeze_splits(manifest, seed)
    manifest_path = output_dir / "unified_cached_manifest.csv"
    split_path = output_dir / "locked_patient_splits.json"
    manifest.to_csv(manifest_path, index=False)
    split_path.write_text(json.dumps(splits, indent=2, sort_keys=True))
    summary = {
        "format": "afib_unified_training_protocol_v1",
        "windows": int(len(manifest)),
        "positives": int(manifest.label.sum()),
        "negatives": int(manifest.label.eq(0).sum()),
        "patients": int(manifest.subject_id.nunique()),
        "source_windows": {
            str(k): int(v) for k, v in manifest.source_family.value_counts().sort_index().items()
        },
        "source_positives": {
            str(k): int(v) for k, v in manifest.groupby("source_family").label.sum().sort_index().items()
        },
        "manifest_sha256": sha256_file(manifest_path),
        "splits_sha256": sha256_file(split_path),
        "split_gates": splits["gates"],
        "all_gates_pass": splits["all_gates_pass"],
        "protected_evaluation": {
            "legacy_locked_final_overwritten": False,
            "cpsc_used_for_training": False,
        },
    }
    (output_dir / "protocol_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True)
    )
    return summary

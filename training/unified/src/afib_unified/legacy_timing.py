"""Recover exact AF timing for cached AFDB/LTAFDB windows from annotations."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .manifest import sha256_file


AF_LABELS = frozenset({"AF", "AFIB"})
NEGATIVE_STRATA = ("near_20_60", "medium_60_360", "far_360_plus_or_none")


def _as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.astype(bool)
    converted = series.astype(str).str.lower().map(
        {"true": True, "false": False, "1": True, "0": False}
    )
    if converted.isna().any():
        raise ValueError("boolean column contains invalid values")
    return converted.astype(bool)


def normalize_rhythm(note: str) -> str | None:
    cleaned = str(note).replace("\x00", "").strip()
    if not cleaned.startswith("("):
        return None
    token = cleaned[1:].split()[0].strip().upper()
    return token or None


def read_af_intervals(record_base: Path) -> tuple[float, np.ndarray, np.ndarray]:
    try:
        import wfdb
    except ImportError as error:
        raise RuntimeError("Install wfdb before enriching legacy timing") from error
    header = wfdb.rdheader(str(record_base))
    annotation = wfdb.rdann(str(record_base), "atr")
    fs = float(header.fs)
    samples = int(header.sig_len)
    rhythm_events = []
    for sample, note in zip(annotation.sample, annotation.aux_note, strict=True):
        label = normalize_rhythm(note)
        if label is not None:
            rhythm_events.append((int(sample), label))
    rhythm_events.sort(key=lambda item: item[0])
    intervals = []
    for index, (start, label) in enumerate(rhythm_events):
        end = rhythm_events[index + 1][0] if index + 1 < len(rhythm_events) else samples
        if end > start and label in AF_LABELS:
            intervals.append((start / fs, end / fs))
    if not intervals:
        return fs, np.empty(0, dtype=float), np.empty(0, dtype=float)
    values = np.asarray(intervals, dtype=float)
    return fs, values[:, 0], values[:, 1]


def _negative_policy(next_minutes: float, since_end_minutes: float) -> tuple[str, bool, str]:
    if np.isfinite(next_minutes) and next_minutes < 20.0 - 1e-6:
        return "INVALID_ONSET_INSIDE_20", False, "negative_has_onset_inside_horizon"
    if np.isfinite(since_end_minutes) and since_end_minutes < 30.0:
        return (
            "post_af_recovery_excluded",
            False,
            "observation_starts_within_30m_after_af_end",
        )
    if np.isfinite(next_minutes) and next_minutes < 60.0:
        return "near_20_60", True, ""
    if np.isfinite(next_minutes) and next_minutes < 360.0:
        return "medium_60_360", True, ""
    return "far_360_plus_or_none", True, ""


def enrich_legacy_timing(
    manifest: pd.DataFrame, dataset_roots: dict[str, Path]
) -> tuple[pd.DataFrame, dict]:
    frame = manifest.sort_values("manifest_index").reset_index(drop=True).copy()
    required = {
        "manifest_index", "cache_family", "cache_record_id", "cache_local_index",
        "dataset", "source_family", "record_id", "subject_id", "label", "window_id",
        "observation_start_seconds", "forecast_start_seconds", "forecast_end_seconds",
        "af_onset_seconds", "negative_stratum", "training_eligible",
    }
    if missing := required - set(frame.columns):
        raise ValueError(f"multi-horizon manifest missing columns: {sorted(missing)}")
    if not np.array_equal(frame.manifest_index.to_numpy(), np.arange(len(frame))):
        raise ValueError("manifest_index must equal row position")
    unknown = set(frame.loc[frame.source_family.eq("legacy"), "dataset"].astype(str)) - set(
        dataset_roots
    )
    if unknown:
        raise ValueError(f"missing dataset roots: {sorted(unknown)}")
    frame["training_eligible"] = _as_bool(frame["training_eligible"])
    frame["evaluation_eligible"] = frame["training_eligible"].astype(bool)
    if "exclusion_reason" not in frame.columns:
        frame["exclusion_reason"] = ""
    frame["legacy_timing_source"] = ""

    failures: list[dict] = []
    positive_mismatches: list[dict] = []
    legacy_mask = frame.source_family.astype(str).eq("legacy")
    grouped = frame.loc[legacy_mask].groupby(["dataset", "record_id"], sort=True)
    for record_number, ((dataset, record_id), rows) in enumerate(grouped, start=1):
        try:
            base = Path(dataset_roots[str(dataset)]) / str(record_id)
            fs, starts, ends = read_af_intervals(base)
            tolerance = max(1.0 / fs, 1.0 / 250.0) + 1e-6
            for index in rows.index:
                row = frame.loc[index]
                observation_start = float(row.observation_start_seconds)
                forecast_start = float(row.forecast_start_seconds)
                future = starts[starts >= forecast_start - tolerance]
                next_seconds = float(future.min()) if len(future) else np.nan
                next_minutes = (
                    max(0.0, (next_seconds - forecast_start) / 60.0)
                    if np.isfinite(next_seconds) else np.nan
                )
                prior = ends[ends <= observation_start + tolerance]
                since_end = (
                    max(0.0, (observation_start - float(prior.max())) / 60.0)
                    if len(prior) else np.nan
                )
                frame.at[index, "next_af_onset_minutes"] = next_minutes
                frame.at[index, "minutes_since_af_end"] = since_end
                frame.at[index, "legacy_timing_source"] = "raw_wfdb_annotations_v1"
                if int(row.label) == 1:
                    expected = float(row.af_onset_seconds)
                    if not np.isfinite(next_seconds) or abs(next_seconds - expected) > tolerance:
                        positive_mismatches.append(
                            {
                                "manifest_index": int(row.manifest_index),
                                "dataset": str(dataset),
                                "record_id": str(record_id),
                                "expected_onset_seconds": expected,
                                "annotation_onset_seconds": next_seconds,
                            }
                        )
                    frame.at[index, "negative_stratum"] = ""
                    frame.at[index, "training_eligible"] = True
                    frame.at[index, "evaluation_eligible"] = True
                    frame.at[index, "exclusion_reason"] = ""
                else:
                    stratum, eligible, reason = _negative_policy(next_minutes, since_end)
                    frame.at[index, "negative_stratum"] = stratum
                    frame.at[index, "training_eligible"] = eligible
                    frame.at[index, "evaluation_eligible"] = eligible
                    frame.at[index, "exclusion_reason"] = reason
            print(
                f"[{record_number}/{len(grouped)}] {dataset}:{record_id}; "
                f"AF intervals={len(starts)}; windows={len(rows)}",
                flush=True,
            )
        except Exception as error:
            failures.append(
                {"dataset": str(dataset), "record_id": str(record_id), "error": repr(error)}
            )
            print(f"FAILED {dataset}:{record_id}: {error}", flush=True)

    negative = frame[legacy_mask & frame.label.astype(int).eq(0)].copy()
    counts = (
        negative.groupby(["outer_test_fold", "negative_stratum"], sort=True)
        .agg(
            windows=("manifest_index", "size"),
            patients=("subject_id", "nunique"),
            eligible_windows=("training_eligible", "sum"),
        )
        .reset_index()
    )
    evidence = {
        "records_audited": int(grouped.ngroups),
        "failed_records": failures,
        "positive_onset_mismatches": positive_mismatches,
        "legacy_negative_counts_by_outer_fold": counts.to_dict("records"),
        "legacy_post_af_recovery_excluded": int(
            negative.negative_stratum.eq("post_af_recovery_excluded").sum()
        ),
        "legacy_eligible_strata": {
            str(key): int(value)
            for key, value in negative.loc[negative.training_eligible]
            .negative_stratum.value_counts().sort_index().items()
        },
    }
    return frame, evidence


def build_and_save_harmonized_manifest(
    *,
    target_dir: Path,
    protocol_dir: Path,
    afdb_root: Path,
    ltafdb_root: Path,
    output_dir: Path,
) -> dict:
    base_path = Path(target_dir) / "multihorizon_cached_manifest.csv"
    target_summary = json.loads((Path(target_dir) / "target_summary.json").read_text())
    protocol_summary = json.loads((Path(protocol_dir) / "protocol_summary.json").read_text())
    if sha256_file(base_path) != target_summary["multihorizon_manifest_sha256"]:
        raise ValueError("multi-horizon manifest hash mismatch")
    if target_summary["cached_manifest_sha256"] != protocol_summary["manifest_sha256"]:
        raise ValueError("target and protocol manifests disagree")
    base = pd.read_csv(
        base_path, dtype={"record_id": str, "subject_id": str, "dataset": str}
    )
    address_columns = [
        "manifest_index", "cache_family", "cache_record_id", "cache_local_index", "window_id"
    ]
    before_addresses = base[address_columns].copy()
    harmonized, evidence = enrich_legacy_timing(
        base, {"afdb": Path(afdb_root), "ltafdb": Path(ltafdb_root)}
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "harmonized_cached_manifest.csv"
    harmonized.to_csv(output_path, index=False)
    negative = harmonized[
        harmonized.source_family.astype(str).eq("legacy")
        & harmonized.label.astype(int).eq(0)
    ]
    eligible_negative = negative[negative.training_eligible.astype(bool)]
    gates = {
        "all_records_readable": not evidence["failed_records"],
        "all_positive_onsets_match": not evidence["positive_onset_mismatches"],
        "all_8292_cache_rows_preserved": len(harmonized) == 8292,
        "cache_addresses_unchanged": before_addresses.equals(harmonized[address_columns]),
        "no_negative_onset_inside_20_minutes": not bool(
            negative.negative_stratum.eq("INVALID_ONSET_INSIDE_20").any()
        ),
        "post_af_recovery_windows_excluded": (
            evidence["legacy_post_af_recovery_excluded"] > 0
            and not bool(
                negative.loc[
                    negative.negative_stratum.eq("post_af_recovery_excluded"),
                    "training_eligible",
                ].astype(bool).any()
            )
        ),
        "all_three_legacy_negative_strata_present": set(
            eligible_negative.negative_stratum.astype(str)
        ) == set(NEGATIVE_STRATA),
        "outer_fold_assignments_preserved": bool(
            harmonized.groupby("subject_id").outer_test_fold.nunique().eq(1).all()
        ),
    }
    summary = {
        "format": "afib_harmonized_legacy_timing_manifest_v1",
        **evidence,
        "windows": int(len(harmonized)),
        "training_eligible_windows": int(harmonized.training_eligible.astype(bool).sum()),
        "evaluation_eligible_windows": int(harmonized.evaluation_eligible.astype(bool).sum()),
        "source_protocol_manifest_sha256": protocol_summary["manifest_sha256"],
        "source_multihorizon_manifest_sha256": target_summary["multihorizon_manifest_sha256"],
        "harmonized_manifest_sha256": sha256_file(output_path),
        "outer_test_evaluated": False,
        "gates": gates,
        "all_gates_pass": all(gates.values()),
    }
    (output_dir / "harmonized_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True)
    )
    if not summary["all_gates_pass"]:
        raise RuntimeError("harmonized legacy timing gates failed")
    return summary


__all__ = [
    "AF_LABELS",
    "NEGATIVE_STRATA",
    "build_and_save_harmonized_manifest",
    "enrich_legacy_timing",
    "normalize_rhythm",
    "read_af_intervals",
]

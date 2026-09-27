"""Audit exact 20-minute AF-forecast windows in selected Icentia records.

This is deliberately annotation-only.  It redownloads only the small WFDB
headers and rhythm annotations for records that contained sampled examples of
both normal rhythm and AF/AFL in the earlier morphology cache.  No ECG signal
files are downloaded.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

from .icentia import (
    ICENTIA_BASE_URL,
    atomic_json,
    digest_file,
    fetch_metadata,
    parse_rhythm_intervals,
)


AUDIT_FORMAT = "icentia_20minute_forecast_annotation_audit_v1"
AF_LABELS = frozenset({"AF", "AFIB", "AFL"})
SINUS_LABEL = "N"
EXPECTED_FS_HZ = 250


@dataclass(frozen=True)
class ForecastWindow:
    patient_id: str
    record: str
    start_sample: int
    end_sample: int
    horizon_end_sample: int
    label: int
    future_af_onset_sample: int | None


@dataclass(frozen=True)
class ForecastWindowAudit:
    accepted_negative: int = 0
    accepted_positive: int = 0
    excluded_non_sinus_observation: int = 0
    excluded_incomplete_horizon: int = 0


def candidate_records(selection: dict) -> list[dict]:
    """Return selected records with sampled examples from both rhythm classes."""
    candidates: list[dict] = []
    seen: set[str] = set()
    for row in selection.get("records", []):
        record = str(row["record"])
        if record in seen:
            raise ValueError(f"duplicate selected record: {record}")
        seen.add(record)
        labels = {int(sample["label"]) for sample in row.get("samples", [])}
        if labels == {0, 1}:
            candidates.append(row)
    return sorted(candidates, key=lambda row: str(row["record"]))


def _coverage(
    intervals: tuple[tuple[int, int, str], ...],
    start: int,
    end: int,
    label: str,
) -> int:
    return sum(
        max(0, min(end, interval_end) - max(start, interval_start))
        for interval_start, interval_end, interval_label in intervals
        if interval_label == label
    )


def enumerate_forecast_windows(
    *,
    patient: str,
    record: str,
    recording_samples: int,
    intervals: tuple[tuple[int, int, str], ...],
    fs_hz: float = EXPECTED_FS_HZ,
    observation_minutes: int = 10,
    horizon_minutes: int = 20,
    stride_minutes: int = 8,
) -> tuple[list[ForecastWindow], ForecastWindowAudit]:
    """Enumerate strict normal-observation, future-onset windows on a fixed grid."""
    if abs(float(fs_hz) - EXPECTED_FS_HZ) > 1e-6:
        raise ValueError(f"expected {EXPECTED_FS_HZ} Hz, found {fs_hz}")
    if min(observation_minutes, horizon_minutes, stride_minutes) < 1:
        raise ValueError("observation, horizon, and stride must be positive")

    minute = int(round(60 * fs_hz))
    observation = observation_minutes * minute
    horizon = horizon_minutes * minute
    stride = stride_minutes * minute
    counts: Counter[str] = Counter()
    windows: list[ForecastWindow] = []

    for start in range(0, max(0, recording_samples - observation + 1), stride):
        end = start + observation
        horizon_end = end + horizon
        if horizon_end > recording_samples:
            counts["excluded_incomplete_horizon"] += 1
            continue
        if _coverage(intervals, start, end, SINUS_LABEL) != observation:
            counts["excluded_non_sinus_observation"] += 1
            continue

        future_onsets = [
            interval_start
            for interval_start, _interval_end, label in intervals
            if label in AF_LABELS and end <= interval_start < horizon_end
        ]
        onset = min(future_onsets) if future_onsets else None
        label = int(onset is not None)
        counts["accepted_positive" if label else "accepted_negative"] += 1
        windows.append(
            ForecastWindow(
                patient_id=patient,
                record=record,
                start_sample=start,
                end_sample=end,
                horizon_end_sample=horizon_end,
                label=label,
                future_af_onset_sample=onset,
            )
        )

    return windows, ForecastWindowAudit(
        **{
            field: int(counts[field])
            for field in ForecastWindowAudit.__dataclass_fields__
        }
    )


def _safe_record_name(record: str) -> str:
    return record.replace("/", "__") + ".json"


def _write_windows(path: Path, windows: list[ForecastWindow]) -> None:
    fields = list(ForecastWindow.__dataclass_fields__)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for window in windows:
            row = asdict(window)
            if row["future_af_onset_sample"] is None:
                row["future_af_onset_sample"] = ""
            writer.writerow(row)
    temporary.replace(path)


def run(args: argparse.Namespace) -> dict:
    selection_path = args.selection.resolve()
    if not selection_path.is_file():
        raise FileNotFoundError(selection_path)
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if selection.get("format") != "icentia_selection_v1":
        raise ValueError("selection is not an Icentia v1 selection.json")

    base_url = args.base_url or selection.get("config", {}).get("base_url")
    if not base_url:
        base_url = ICENTIA_BASE_URL
    candidates = candidate_records(selection)
    if not candidates:
        raise RuntimeError("selection contains no records with both sampled classes")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    record_dir = args.output_dir / "record_audits"
    record_dir.mkdir(exist_ok=True)
    config = {
        "format": AUDIT_FORMAT,
        "selection_sha256": digest_file(selection_path),
        "base_url": base_url,
        "candidate_policy": "selected records containing sampled normal and AF/AFL",
        "observation_minutes": args.observation_minutes,
        "horizon_minutes": args.horizon_minutes,
        "stride_minutes": args.stride_minutes,
    }
    config_path = args.output_dir / "config.json"
    if config_path.is_file():
        if json.loads(config_path.read_text(encoding="utf-8")) != config:
            raise ValueError("audit configuration changed; use a new output directory")
    else:
        atomic_json(config_path, config)

    all_windows: list[ForecastWindow] = []
    results: list[dict] = []
    failures: list[dict[str, str]] = []
    started = time.monotonic()
    for index, row in enumerate(candidates, start=1):
        record = str(row["record"])
        patient = str(row["patient_id"])
        saved_path = record_dir / _safe_record_name(record)
        try:
            if saved_path.is_file():
                saved = json.loads(saved_path.read_text(encoding="utf-8"))
                if saved.get("config") != config or saved.get("record") != record:
                    raise ValueError(f"saved record audit has incompatible configuration: {saved_path}")
                record_windows = [ForecastWindow(**item) for item in saved["windows"]]
                result = saved["result"]
                print(f"RESUME audit {index}/{len(candidates)}: {record}", flush=True)
            else:
                local = fetch_metadata(args.output_dir, base_url, record)
                fs_hz, signal_samples, intervals = parse_rhythm_intervals(local)
                record_windows, audit = enumerate_forecast_windows(
                    patient=patient,
                    record=record,
                    recording_samples=signal_samples,
                    intervals=intervals,
                    fs_hz=fs_hz,
                    observation_minutes=args.observation_minutes,
                    horizon_minutes=args.horizon_minutes,
                    stride_minutes=args.stride_minutes,
                )
                result = {
                    "patient_id": patient,
                    "record": record,
                    "signal_samples": signal_samples,
                    "interval_count": len(intervals),
                    "audit": asdict(audit),
                }
                atomic_json(
                    saved_path,
                    {
                        "config": config,
                        "record": record,
                        "result": result,
                        "windows": [asdict(window) for window in record_windows],
                    },
                )
            all_windows.extend(record_windows)
            results.append(result)
        except Exception as error:
            failures.append({"patient_id": patient, "record": record, "reason": repr(error)})
            print(f"AUDIT FAILURE {record}: {error}", flush=True)

        elapsed = time.monotonic() - started
        eta = elapsed / index * (len(candidates) - index)
        positives = sum(window.label for window in all_windows)
        positive_patients = len(
            {window.patient_id for window in all_windows if window.label == 1}
        )
        print(
            f"AUDIT record {index}/{len(candidates)}; positives={positives}; "
            f"positive_patients={positive_patients}; failures={len(failures)}; "
            f"elapsed={elapsed / 60:.1f} min; ETA={eta / 60:.1f} min",
            flush=True,
        )
        atomic_json(
            args.output_dir / "status.json",
            {
                "phase": "annotation_audit",
                "records_completed": index,
                "records_total": len(candidates),
                "positive_windows_so_far": positives,
                "positive_patients_so_far": positive_patients,
                "failures": len(failures),
                "elapsed_seconds": elapsed,
                "eta_seconds": eta,
            },
        )

    _write_windows(args.output_dir / "windows.csv", all_windows)
    positive_windows = [window for window in all_windows if window.label == 1]
    negative_windows = [window for window in all_windows if window.label == 0]
    positive_patients = sorted({window.patient_id for window in positive_windows})
    positive_records = sorted({window.record for window in positive_windows})
    positives_per_patient = Counter(window.patient_id for window in positive_windows)
    capped_positive_windows = sum(min(count, 4) for count in positives_per_patient.values())
    audit_totals = {
        field: sum(int(result["audit"][field]) for result in results)
        for field in ForecastWindowAudit.__dataclass_fields__
    }
    gate = (
        not failures
        and len(positive_patients) >= 10
        and capped_positive_windows >= 20
    )
    summary = {
        "format": AUDIT_FORMAT,
        "all_records_complete": not failures,
        "candidate_records": len(candidates),
        "records_audited": len(results),
        "failed_records": failures,
        "accepted_windows": len(all_windows),
        "positive_windows": len(positive_windows),
        "negative_windows": len(negative_windows),
        "positive_records": len(positive_records),
        "positive_patients": len(positive_patients),
        "positive_patient_ids": positive_patients,
        "positive_windows_by_patient": dict(sorted(positives_per_patient.items())),
        "positive_windows_capped_at_4_per_patient": capped_positive_windows,
        "window_audit_totals": audit_totals,
        "aligned_pretraining_gate": gate,
        "gate_rule": (
            "all candidate audits complete; at least 10 positive patients and "
            "at least 20 positive windows after a four-per-patient cap"
        ),
        "config": config,
    }
    atomic_json(args.output_dir / "summary.json", summary)
    if failures:
        print(json.dumps(summary, indent=2), flush=True)
        raise RuntimeError(
            f"{len(failures)} record audits failed; rerun the identical command to resume"
        )
    (args.output_dir / "COMPLETE").write_text("ok\n", encoding="utf-8")
    atomic_json(args.output_dir / "status.json", {"phase": "complete", **summary})
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--base-url")
    parser.add_argument("--observation-minutes", type=int, default=10)
    parser.add_argument("--horizon-minutes", type=int, default=20)
    parser.add_argument("--stride-minutes", type=int, default=8)
    args = parser.parse_args(argv)
    if min(args.observation_minutes, args.horizon_minutes, args.stride_minutes) < 1:
        parser.error("observation, horizon, and stride must be positive")
    run(args)


if __name__ == "__main__":
    main()

"""Build an auditable AFDB/LTAFDB window and outer-fold manifest."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from .contracts import DEFAULT_CONTRACT
from .rhythm import RhythmInterval, build_rhythm_intervals
from .splits import make_outer_group_folds
from .windowing import WindowAudit, WindowRecord, enumerate_windows


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    directory: Path
    expected_fs_hz: float
    annotation_extension: str = "atr"
    lead_index: int = 0
    metadata_directory: Path | None = None

    @property
    def metadata_root(self) -> Path:
        return self.metadata_directory or self.directory


@dataclass(frozen=True)
class RecordManifest:
    dataset: str
    record_id: str
    subject_id: str
    record_path: str
    source_fs_hz: float
    source_samples: int
    target_samples: int
    lead_index: int
    lead_name: str
    windows: tuple[WindowRecord, ...]
    audit: WindowAudit


def scale_sample(sample: int, fs_in_hz: float, fs_out_hz: float) -> int:
    return round(int(sample) * fs_out_hz / fs_in_hz)


def scale_rhythm_intervals(
    intervals: Iterable[RhythmInterval], fs_in_hz: float, fs_out_hz: float
) -> tuple[RhythmInterval, ...]:
    scaled: list[RhythmInterval] = []
    for interval in intervals:
        start = scale_sample(interval.start_sample, fs_in_hz, fs_out_hz)
        end = scale_sample(interval.end_sample, fs_in_hz, fs_out_hz)
        if end > start:
            scaled.append(RhythmInterval(start, end, interval.label))
    return tuple(scaled)


def _metadata_record_names(directory: Path) -> tuple[str, ...]:
    headers = {path.stem for path in directory.glob("*.hea")}
    annotations = {path.stem for path in directory.glob("*.atr")}
    return tuple(sorted(headers & annotations))


def process_record(spec: DatasetSpec, record_id: str) -> RecordManifest:
    try:
        import wfdb
    except ImportError as error:
        raise RuntimeError("Install the project dependencies before building a manifest") from error

    record_path = spec.directory / record_id
    metadata_path = spec.metadata_root / record_id
    header = wfdb.rdheader(str(metadata_path))
    source_fs = float(header.fs)
    source_samples = int(header.sig_len)
    if source_samples <= 0 or int(header.n_sig) <= spec.lead_index:
        raise ValueError("record has no usable signal")
    if abs(source_fs - spec.expected_fs_hz) > 1e-6:
        raise ValueError(
            f"unexpected sampling frequency {source_fs}; expected {spec.expected_fs_hz}"
        )

    annotation = wfdb.rdann(str(metadata_path), spec.annotation_extension)
    native_rhythms = build_rhythm_intervals(
        annotation.sample,
        annotation.aux_note,
        recording_samples=source_samples,
    )
    target_rhythms = scale_rhythm_intervals(
        native_rhythms, source_fs, DEFAULT_CONTRACT.target_fs_hz
    )
    target_samples = scale_sample(source_samples, source_fs, DEFAULT_CONTRACT.target_fs_hz)
    subject_id = f"{spec.name}:{record_id}"
    windows, audit = enumerate_windows(
        recording_samples=target_samples,
        rhythm_intervals=target_rhythms,
        dataset=spec.name,
        record_id=record_id,
        subject_id=subject_id,
        contract=DEFAULT_CONTRACT,
        require_annotated_sinus=True,
        sinus_labels=frozenset({"N"}),
    )
    lead_names = list(header.sig_name or [])
    lead_name = lead_names[spec.lead_index] if lead_names else f"lead_{spec.lead_index}"
    return RecordManifest(
        dataset=spec.name,
        record_id=record_id,
        subject_id=subject_id,
        record_path=str(record_path),
        source_fs_hz=source_fs,
        source_samples=source_samples,
        target_samples=target_samples,
        lead_index=spec.lead_index,
        lead_name=lead_name,
        windows=windows,
        audit=audit,
    )


def build_manifest(specs: Iterable[DatasetSpec]) -> tuple[list[RecordManifest], list[dict]]:
    records: list[RecordManifest] = []
    skipped: list[dict] = []
    for spec in specs:
        if not spec.directory.is_dir():
            raise FileNotFoundError(f"dataset directory not found: {spec.directory}")
        if not spec.metadata_root.is_dir():
            raise FileNotFoundError(f"metadata directory not found: {spec.metadata_root}")
        names = _metadata_record_names(spec.metadata_root)
        print(f"{spec.name}: found {len(names)} .hea/.atr metadata pairs")
        for index, record_id in enumerate(names, start=1):
            try:
                result = process_record(spec, record_id)
                records.append(result)
                positives = sum(window.label for window in result.windows)
                print(
                    f"[{spec.name}] {index}/{len(names)} {record_id}: "
                    f"windows={len(result.windows)}, positive={positives}"
                )
            except Exception as error:
                skipped.append(
                    {"dataset": spec.name, "record_id": record_id, "reason": repr(error)}
                )
                print(f"[{spec.name}] SKIP {record_id}: {error}")
    return records, skipped


def _write_csv(path: Path, records: list[RecordManifest]) -> list[WindowRecord]:
    fieldnames = [
        "dataset",
        "record_id",
        "subject_id",
        "record_path",
        "source_fs_hz",
        "lead_index",
        "lead_name",
        "start_sample_250hz",
        "end_sample_250hz",
        "horizon_end_sample_250hz",
        "start_seconds",
        "end_seconds",
        "label",
        "future_af_onset_sample_250hz",
    ]
    all_windows: list[WindowRecord] = []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            for window in record.windows:
                all_windows.append(window)
                writer.writerow(
                    {
                        "dataset": record.dataset,
                        "record_id": record.record_id,
                        "subject_id": record.subject_id,
                        "record_path": record.record_path,
                        "source_fs_hz": record.source_fs_hz,
                        "lead_index": record.lead_index,
                        "lead_name": record.lead_name,
                        "start_sample_250hz": window.start_sample,
                        "end_sample_250hz": window.end_sample,
                        "horizon_end_sample_250hz": window.horizon_end_sample,
                        "start_seconds": window.start_sample / DEFAULT_CONTRACT.target_fs_hz,
                        "end_seconds": window.end_sample / DEFAULT_CONTRACT.target_fs_hz,
                        "label": window.label,
                        "future_af_onset_sample_250hz": (
                            ""
                            if window.future_af_onset_sample is None
                            else window.future_af_onset_sample
                        ),
                    }
                )
    return all_windows


def _class_counts(windows: Iterable[WindowRecord]) -> dict[str, int]:
    labels = [window.label for window in windows]
    return {
        "total": len(labels),
        "negative": labels.count(0),
        "positive": labels.count(1),
    }


def write_outputs(
    output_dir: Path, records: list[RecordManifest], skipped: list[dict]
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    windows = _write_csv(output_dir / "windows.csv", records)
    if not windows:
        raise RuntimeError("manifest contains no accepted windows")

    folds = make_outer_group_folds(windows, n_splits=5)
    fold_payload = []
    for fold in folds:
        train_windows = [windows[index] for index in fold.train_indices]
        test_windows = [windows[index] for index in fold.test_indices]
        test_counts = _class_counts(test_windows)
        if test_counts["positive"] == 0 or test_counts["negative"] == 0:
            raise RuntimeError(f"outer fold {fold.fold} does not contain both classes")
        fold_payload.append(
            {
                "fold": fold.fold,
                "train_subjects": list(fold.train_subjects),
                "test_subjects": list(fold.test_subjects),
                "train_counts": _class_counts(train_windows),
                "test_counts": test_counts,
            }
        )
    (output_dir / "folds.json").write_text(json.dumps(fold_payload, indent=2))

    datasets: dict[str, dict[str, int]] = {}
    for name in sorted({record.dataset for record in records}):
        selected = [window for record in records if record.dataset == name for window in record.windows]
        datasets[name] = _class_counts(selected)

    audit_totals = {
        field: sum(getattr(record.audit, field) for record in records)
        for field in WindowAudit.__dataclass_fields__
    }
    summary = {
        "contract": asdict(DEFAULT_CONTRACT),
        "sinus_labels": ["N"],
        "subject_identity": "dataset_name:record_id",
        "record_count": len(records),
        "skipped_records": skipped,
        "windows": _class_counts(windows),
        "windows_by_dataset": datasets,
        "window_audit_totals": audit_totals,
        "outer_folds": fold_payload,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--afdb", required=True, type=Path)
    parser.add_argument("--ltafdb", required=True, type=Path)
    parser.add_argument("--afdb-metadata", type=Path)
    parser.add_argument("--ltafdb-metadata", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    specs = (
        DatasetSpec(
            "afdb", args.afdb, expected_fs_hz=250.0,
            metadata_directory=args.afdb_metadata,
        ),
        DatasetSpec(
            "ltafdb", args.ltafdb, expected_fs_hz=128.0,
            metadata_directory=args.ltafdb_metadata,
        ),
    )
    records, skipped = build_manifest(specs)
    summary = write_outputs(args.output_dir, records, skipped)
    print(json.dumps(summary["windows_by_dataset"], indent=2))
    print(f"Saved manifest artifacts to {args.output_dir}")


if __name__ == "__main__":
    main()

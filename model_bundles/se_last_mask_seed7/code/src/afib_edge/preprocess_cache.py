"""Resumable record-wise ECG and HRV cache generation from Manifest v1."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Callable, Iterator

import numpy as np

from .contracts import DEFAULT_CONTRACT, HRV_FEATURE_NAMES
from .hrv import compute_hrv_features
from .signal import bandpass_filter, resample_ecg


def _sha256(path: Path, chunk_bytes: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def read_manifest(path: Path) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            grouped[row["subject_id"]].append(row)
    if not grouped:
        raise ValueError("window manifest is empty")
    for rows in grouped.values():
        rows.sort(key=lambda row: int(row["start_sample_250hz"]))
    return dict(sorted(grouped.items()))


@contextmanager
def stage_wfdb_record(record_path: Path, stage_root: Path) -> Iterator[Path]:
    """Copy one WFDB signal/header pair locally to avoid random Drive reads."""
    stage_root.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=f"{record_path.name}_", dir=stage_root) as directory:
        staged_root = Path(directory)
        for extension in ("hea", "dat"):
            source = record_path.with_suffix(f".{extension}")
            if not source.is_file():
                raise FileNotFoundError(source)
            shutil.copy2(source, staged_root / source.name)
        yield staged_root / record_path.name


def load_and_prepare_signal(staged_record_path: Path, lead_index: int) -> np.ndarray:
    try:
        import wfdb
    except ImportError as error:
        raise RuntimeError("Install project dependencies before preprocessing") from error

    record = wfdb.rdrecord(str(staged_record_path), channels=[lead_index], physical=True)
    signal = np.asarray(record.p_signal[:, 0], dtype=np.float64)
    if not np.isfinite(signal).all():
        raise ValueError(f"non-finite ECG samples in {staged_record_path.name}")
    filtered = bandpass_filter(signal, float(record.fs))
    return resample_ecg(filtered, float(record.fs), DEFAULT_CONTRACT.target_fs_hz)


XQRS_CHUNK_SECONDS = 30 * 60
XQRS_OVERLAP_SECONDS = 15
RPEAK_DETECTOR_NAME = "wfdb.xqrs chunked (30-minute cores, 15-second overlap)"


def _chunked_rpeak_detection(
    signal: np.ndarray,
    fs_hz: float,
    detector: Callable[..., np.ndarray],
    chunk_seconds: int = XQRS_CHUNK_SECONDS,
    overlap_seconds: int = XQRS_OVERLAP_SECONDS,
) -> np.ndarray:
    """Run a detector on overlapping chunks and keep each chunk's core peaks.

    Long noisy Holter records can make XQRS backsearch pathologically slow.
    Fixed cores bound that work, while overlap supplies context on both sides of
    internal boundaries. Each sample belongs to exactly one retained core, so
    overlap detections cannot create duplicate beats.
    """
    x = np.asarray(signal, dtype=np.float64)
    if x.ndim != 1 or len(x) == 0:
        raise ValueError("ECG signal must be a non-empty one-dimensional array")
    if fs_hz <= 0 or chunk_seconds <= 0 or overlap_seconds < 0:
        raise ValueError("sampling rate/chunk must be positive and overlap non-negative")

    core_samples = max(1, int(round(chunk_seconds * fs_hz)))
    overlap_samples = int(round(overlap_seconds * fs_hz))
    core_starts = list(range(0, len(x), core_samples))
    retained: list[np.ndarray] = []

    for chunk_index, core_start in enumerate(core_starts, start=1):
        core_end = min(core_start + core_samples, len(x))
        detect_start = max(0, core_start - overlap_samples)
        detect_end = min(len(x), core_end + overlap_samples)
        print(
            f"    XQRS chunk {chunk_index}/{len(core_starts)} "
            f"({detect_start / fs_hz / 3600:.2f}-{detect_end / fs_hz / 3600:.2f} h)",
            flush=True,
        )
        local = np.asarray(
            detector(sig=x[detect_start:detect_end], fs=fs_hz, verbose=False),
            dtype=np.int64,
        )
        global_peaks = local + detect_start
        retained.append(
            global_peaks[(global_peaks >= core_start) & (global_peaks < core_end)]
        )

    return np.unique(np.concatenate(retained)) if retained else np.empty(0, dtype=np.int64)


def detect_record_rpeaks(signal_250hz: np.ndarray) -> np.ndarray:
    try:
        from wfdb.processing import xqrs_detect
    except ImportError as error:
        raise RuntimeError("WFDB XQRS is required for reproducible R-peak detection") from error
    peaks = _chunked_rpeak_detection(
        signal_250hz,
        DEFAULT_CONTRACT.target_fs_hz,
        xqrs_detect,
    )
    peaks = np.asarray(peaks, dtype=np.int64)
    if len(peaks) < 3 or np.any(np.diff(peaks) <= 0):
        raise ValueError("R-peak detector returned an invalid sequence")
    return peaks


def segment_hrv(
    rpeaks: np.ndarray, segment_start: int, segment_end: int
) -> tuple[np.ndarray, int, int]:
    left = int(np.searchsorted(rpeaks, segment_start, side="left"))
    right = int(np.searchsorted(rpeaks, segment_end, side="left"))
    local_peaks = rpeaks[left:right] - segment_start
    result = compute_hrv_features(local_peaks, DEFAULT_CONTRACT.target_fs_hz)
    return result.values, result.valid_rr_count, result.rejected_rr_count


def extract_record_arrays(
    signal: np.ndarray,
    rpeaks: np.ndarray,
    rows: list[dict[str, str]],
    ecg_output: Path,
    hrv_output: Path,
) -> dict[str, object]:
    """Write exact-contract arrays incrementally using NumPy memory maps."""
    count = len(rows)
    ecg = np.lib.format.open_memmap(
        ecg_output,
        mode="w+",
        dtype=np.float32,
        shape=(count, DEFAULT_CONTRACT.segments_per_window, DEFAULT_CONTRACT.segment_samples),
    )
    hrv = np.lib.format.open_memmap(
        hrv_output,
        mode="w+",
        dtype=np.float32,
        shape=(count, DEFAULT_CONTRACT.segments_per_window, len(HRV_FEATURE_NAMES)),
    )
    valid_rr_total = 0
    rejected_rr_total = 0
    for window_index, row in enumerate(rows):
        start = int(row["start_sample_250hz"])
        end = int(row["end_sample_250hz"])
        if end - start != DEFAULT_CONTRACT.observation_samples or end > len(signal):
            raise ValueError(f"invalid window bounds for {row['subject_id']} at {start}")
        for segment_index in range(DEFAULT_CONTRACT.segments_per_window):
            segment_start = start + segment_index * DEFAULT_CONTRACT.segment_samples
            segment_end = segment_start + DEFAULT_CONTRACT.segment_samples
            segment = signal[segment_start:segment_end]
            if segment.shape != (DEFAULT_CONTRACT.segment_samples,):
                raise ValueError("segment violates the fixed 7,500-sample contract")
            ecg[window_index, segment_index] = segment
            values, valid_rr, rejected_rr = segment_hrv(rpeaks, segment_start, segment_end)
            hrv[window_index, segment_index] = values
            valid_rr_total += valid_rr
            rejected_rr_total += rejected_rr
    ecg.flush()
    hrv.flush()
    del ecg, hrv

    hrv_check = np.load(hrv_output, mmap_mode="r")
    finite_by_feature = np.isfinite(hrv_check).sum(axis=(0, 1)).astype(int).tolist()
    total_segments = count * DEFAULT_CONTRACT.segments_per_window
    return {
        "window_count": count,
        "positive_count": sum(int(row["label"]) for row in rows),
        "ecg_shape": [count, 20, 7500],
        "hrv_shape": [count, 20, 6],
        "hrv_feature_order": list(HRV_FEATURE_NAMES),
        "hrv_finite_count_by_feature": finite_by_feature,
        "hrv_missing_count_by_feature": [total_segments - value for value in finite_by_feature],
        "valid_rr_interval_count": valid_rr_total,
        "rejected_rr_interval_count": rejected_rr_total,
    }


def process_subject(
    subject_id: str,
    rows: list[dict[str, str]],
    output_root: Path,
    stage_root: Path,
    resume: bool,
) -> dict[str, object]:
    safe_name = subject_id.replace(":", "__")
    final_dir = output_root / safe_name
    complete_marker = final_dir / "COMPLETE"
    if resume and complete_marker.is_file():
        return json.loads((final_dir / "metadata.json").read_text())
    if final_dir.exists():
        raise FileExistsError(f"incomplete cache already exists: {final_dir}")

    record_path = Path(rows[0]["record_path"])
    lead_index = int(rows[0]["lead_index"])
    with TemporaryDirectory(prefix=f".{safe_name}_", dir=output_root) as temporary:
        temporary_dir = Path(temporary)
        with stage_wfdb_record(record_path, stage_root) as staged_record:
            signal = load_and_prepare_signal(staged_record, lead_index)
        rpeaks = detect_record_rpeaks(signal)
        metadata = extract_record_arrays(
            signal,
            rpeaks,
            rows,
            temporary_dir / "ecg.npy",
            temporary_dir / "hrv.npy",
        )
        labels = np.asarray([int(row["label"]) for row in rows], dtype=np.uint8)
        starts = np.asarray([int(row["start_sample_250hz"]) for row in rows], dtype=np.int64)
        np.save(temporary_dir / "labels.npy", labels)
        np.save(temporary_dir / "starts.npy", starts)
        metadata.update(
            {
                "subject_id": subject_id,
                "dataset": rows[0]["dataset"],
                "record_id": rows[0]["record_id"],
                "record_path": str(record_path),
                "source_fs_hz": float(rows[0]["source_fs_hz"]),
                "lead_index": lead_index,
                "lead_name": rows[0]["lead_name"],
                "detector": RPEAK_DETECTOR_NAME,
                "filter": "Butterworth SOS zero-phase 0.5-40 Hz order 4",
                "resampler": "scipy.signal.resample_poly to 250 Hz",
                "rpeak_count": int(len(rpeaks)),
            }
        )
        for filename in ("ecg.npy", "hrv.npy", "labels.npy", "starts.npy"):
            metadata.setdefault("sha256", {})[filename] = _sha256(temporary_dir / filename)
        (temporary_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
        (temporary_dir / "COMPLETE").write_text("ok\n")
        Path(temporary).rename(final_dir)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--stage-dir", type=Path, default=Path("/content/wfdb_stage"))
    parser.add_argument("--limit-records", type=int)
    parser.add_argument(
        "--subject-id",
        help="Process one exact manifest subject (for example, ltafdb:117)",
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    grouped = read_manifest(args.manifest)
    items = list(grouped.items())
    if args.subject_id is not None:
        if args.subject_id not in grouped:
            raise ValueError(f"subject not found in manifest: {args.subject_id}")
        items = [(args.subject_id, grouped[args.subject_id])]
    if args.limit_records is not None:
        if args.limit_records <= 0:
            raise ValueError("--limit-records must be positive")
        items = items[: args.limit_records]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    run_results = []
    for index, (subject_id, rows) in enumerate(items, start=1):
        print(f"[{index}/{len(items)}] preprocessing {subject_id} ({len(rows)} windows)")
        metadata = process_subject(
            subject_id, rows, args.output_dir, args.stage_dir, args.resume
        )
        run_results.append(metadata)
        print(f"  complete; HRV missing by feature={metadata['hrv_missing_count_by_feature']}")

    run_summary = {
        "manifest": str(args.manifest),
        "record_count": len(run_results),
        "window_count": sum(int(item["window_count"]) for item in run_results),
        "positive_count": sum(int(item["positive_count"]) for item in run_results),
        "records": run_results,
    }
    (args.output_dir / "preprocessing_summary.json").write_text(
        json.dumps(run_summary, indent=2)
    )
    print(json.dumps({k: v for k, v in run_summary.items() if k != "records"}, indent=2))


if __name__ == "__main__":
    main()

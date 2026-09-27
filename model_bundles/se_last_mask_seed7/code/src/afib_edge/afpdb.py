"""Build a resumable, patient-paired AFPDB sequence-pretraining cache.

Only the labelled ``p*`` learning records are used.  Each odd/even record pair
belongs to one subject: the odd record is PAF-distant and the even record is
immediately pre-PAF.  Two non-overlapping ten-minute observations are retained
from the final twenty minutes of each 30-minute record.  Raw signal files are
staged locally and discarded after each pair is processed.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from .contracts import DEFAULT_CONTRACT, HRV_FEATURE_NAMES
from .icentia import atomic_json, download, record_url
from .preprocess_cache import _sha256, detect_record_rpeaks, segment_hrv
from .signal import bandpass_filter, resample_ecg


AFPDB_BASE_URL = "https://physionet.org/files/afpdb/1.0.0"
CACHE_FORMAT = "afpdb_sequence_pretraining_cache_v1"
EXPECTED_FS_HZ = 128.0
WINDOW_START_MINUTES_FROM_RECORD_START = (10, 20)


@dataclass(frozen=True)
class RecordPair:
    subject_id: str
    distant_record: str
    pre_paf_record: str


def parse_learning_pairs(record_names: list[str]) -> list[RecordPair]:
    """Parse complete odd/even ``p`` record pairs from PhysioNet RECORDS."""
    numbered: dict[int, str] = {}
    for raw in record_names:
        name = Path(raw.strip()).name
        match = re.fullmatch(r"p(\d{2})", name)
        if not match:
            continue
        number = int(match.group(1))
        if number in numbered:
            raise ValueError(f"duplicate AFPDB learning record number: {number}")
        numbered[number] = raw.strip()

    pairs: list[RecordPair] = []
    for odd in sorted(number for number in numbered if number % 2 == 1):
        even = odd + 1
        if even not in numbered:
            raise ValueError(f"AFPDB pair is incomplete: p{odd:02d}/p{even:02d}")
        pairs.append(
            RecordPair(
                subject_id=f"afpdb:p{odd:02d}-p{even:02d}",
                distant_record=numbered[odd],
                pre_paf_record=numbered[even],
            )
        )
    if not 20 <= len(pairs) <= 30:
        raise RuntimeError(f"expected 20-30 labelled PAF subject pairs, found {len(pairs)}")
    return pairs


def end_aligned_window_starts(signal_samples: int, fs_hz: int = 250) -> tuple[int, int]:
    """Return adjacent 10-minute windows covering the final 20 minutes."""
    observation = 10 * 60 * fs_hz
    required = 30 * 60 * fs_hz
    if signal_samples < required:
        raise ValueError("AFPDB record is shorter than the documented 30 minutes")
    end = signal_samples
    starts = (end - 2 * observation, end - observation)
    if starts[0] < 0 or starts[1] - starts[0] != observation:
        raise AssertionError("invalid AFPDB end-aligned window geometry")
    return starts


def fetch_record_inventory(output_dir: Path, base_url: str) -> list[str]:
    path = download(f"{base_url.rstrip('/')}/RECORDS", output_dir / "SOURCE_RECORDS")
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]


def _load_record(stage: Path, base_url: str, record_name: str) -> np.ndarray:
    basename = Path(record_name).name
    base = stage / basename
    download(record_url(base_url, record_name, "hea"), base.with_suffix(".hea"))
    download(record_url(base_url, record_name, "dat"), base.with_suffix(".dat"))
    try:
        import wfdb
    except ImportError as error:
        raise RuntimeError("Install wfdb before building the AFPDB cache") from error

    record = wfdb.rdrecord(str(base), channels=[0], physical=True)
    if abs(float(record.fs) - EXPECTED_FS_HZ) > 1e-6:
        raise ValueError(f"unexpected AFPDB sampling rate: {record.fs}")
    signal = np.asarray(record.p_signal[:, 0], dtype=np.float64)
    if not np.isfinite(signal).all():
        raise ValueError(f"non-finite signal samples in {record_name}")
    filtered = bandpass_filter(signal, float(record.fs))
    return resample_ecg(filtered, float(record.fs), DEFAULT_CONTRACT.target_fs_hz)


def _extract_windows(
    signal: np.ndarray,
    rpeaks: np.ndarray,
    label: int,
) -> tuple[list[np.ndarray], list[np.ndarray], list[int], dict[str, int]]:
    ecg_rows: list[np.ndarray] = []
    hrv_rows: list[np.ndarray] = []
    starts_out: list[int] = []
    valid_rr_total = 0
    rejected_rr_total = 0
    for start in end_aligned_window_starts(len(signal)):
        ecg = np.empty(
            (DEFAULT_CONTRACT.segments_per_window, DEFAULT_CONTRACT.segment_samples),
            dtype=np.float32,
        )
        hrv = np.empty(
            (DEFAULT_CONTRACT.segments_per_window, len(HRV_FEATURE_NAMES)),
            dtype=np.float32,
        )
        for segment_index in range(DEFAULT_CONTRACT.segments_per_window):
            segment_start = start + segment_index * DEFAULT_CONTRACT.segment_samples
            segment_end = segment_start + DEFAULT_CONTRACT.segment_samples
            ecg[segment_index] = signal[segment_start:segment_end]
            values, valid_rr, rejected_rr = segment_hrv(
                rpeaks, segment_start, segment_end
            )
            hrv[segment_index] = values
            valid_rr_total += valid_rr
            rejected_rr_total += rejected_rr
        ecg_rows.append(ecg)
        hrv_rows.append(hrv)
        starts_out.append(start)
    return ecg_rows, hrv_rows, starts_out, {
        "label": int(label),
        "valid_rr_intervals": valid_rr_total,
        "rejected_rr_intervals": rejected_rr_total,
    }


def process_pair(
    pair: RecordPair,
    output_dir: Path,
    stage_dir: Path,
    base_url: str,
    resume: bool,
) -> dict:
    safe_name = pair.subject_id.replace(":", "__")
    final_dir = output_dir / safe_name
    if resume and (final_dir / "COMPLETE").is_file():
        return json.loads((final_dir / "metadata.json").read_text(encoding="utf-8"))
    if final_dir.exists():
        raise FileExistsError(f"incomplete AFPDB pair cache exists: {final_dir}")

    stage_dir.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="afpdb_raw_", dir=stage_dir) as raw_temp:
        raw_root = Path(raw_temp)
        arrays: list[tuple[str, int, list[np.ndarray], list[np.ndarray], list[int], dict]] = []
        for record_name, label in (
            (pair.distant_record, 0),
            (pair.pre_paf_record, 1),
        ):
            record_stage = raw_root / Path(record_name).name
            record_stage.mkdir()
            signal = _load_record(record_stage, base_url, record_name)
            rpeaks = detect_record_rpeaks(signal)
            ecg, hrv, starts, counts = _extract_windows(signal, rpeaks, label)
            counts["rpeak_count"] = int(len(rpeaks))
            arrays.append((record_name, label, ecg, hrv, starts, counts))

    with TemporaryDirectory(prefix=f".{safe_name}_", dir=output_dir) as temporary:
        target = Path(temporary)
        ecg_rows, hrv_rows, labels, starts, record_ids = [], [], [], [], []
        record_summaries = []
        for record_name, label, ecg, hrv, record_starts, counts in arrays:
            ecg_rows.extend(ecg)
            hrv_rows.extend(hrv)
            labels.extend([label] * len(ecg))
            starts.extend(record_starts)
            record_ids.extend([Path(record_name).name] * len(ecg))
            record_summaries.append({"record": record_name, **counts})
        order = sorted(range(len(labels)), key=lambda index: (starts[index], record_ids[index]))
        ecg_rows = [ecg_rows[index] for index in order]
        hrv_rows = [hrv_rows[index] for index in order]
        labels = [labels[index] for index in order]
        starts = [starts[index] for index in order]
        record_ids = [record_ids[index] for index in order]
        np.save(target / "ecg.npy", np.stack(ecg_rows).astype(np.float32))
        np.save(target / "hrv.npy", np.stack(hrv_rows).astype(np.float32))
        np.save(target / "labels.npy", np.asarray(labels, dtype=np.uint8))
        np.save(target / "starts.npy", np.asarray(starts, dtype=np.int64))
        np.save(target / "record_ids.npy", np.asarray(record_ids))

        hrv_array = np.asarray(hrv_rows)
        finite = np.isfinite(hrv_array).sum(axis=(0, 1)).astype(int)
        total_segments = len(labels) * DEFAULT_CONTRACT.segments_per_window
        metadata = {
            "format": CACHE_FORMAT,
            "subject_id": pair.subject_id,
            "dataset": "afpdb",
            "record_id": f"{Path(pair.distant_record).name}-{Path(pair.pre_paf_record).name}",
            "window_count": len(labels),
            "positive_count": int(sum(labels)),
            "ecg_shape": list(np.asarray(ecg_rows).shape),
            "hrv_shape": list(hrv_array.shape),
            "hrv_feature_order": list(HRV_FEATURE_NAMES),
            "hrv_missing_count_by_feature": (total_segments - finite).tolist(),
            "records": record_summaries,
            "window_policy": "two adjacent 10-minute windows covering final 20 minutes",
            "lead_index": 0,
            "source_fs_hz": EXPECTED_FS_HZ,
            "target_fs_hz": DEFAULT_CONTRACT.target_fs_hz,
            "sha256": {},
        }
        for filename in ("ecg.npy", "hrv.npy", "labels.npy", "starts.npy", "record_ids.npy"):
            metadata["sha256"][filename] = _sha256(target / filename)
        atomic_json(target / "metadata.json", metadata)
        (target / "COMPLETE").write_text("ok\n", encoding="utf-8")
        target.rename(final_dir)
    return metadata


def _write_manifest(path: Path, pairs: list[RecordPair], output_dir: Path) -> list[dict]:
    rows: list[dict] = []
    for pair in pairs:
        safe_name = pair.subject_id.replace(":", "__")
        record_dir = output_dir / safe_name
        labels = np.load(record_dir / "labels.npy")
        starts = np.load(record_dir / "starts.npy")
        record_ids = np.load(record_dir / "record_ids.npy")
        for label, start, record_id in zip(labels, starts, record_ids, strict=True):
            rows.append(
                {
                    "dataset": "afpdb",
                    "record_id": str(record_id),
                    "subject_id": pair.subject_id,
                    "record_path": "staged_from_physionet_and_discarded",
                    "source_fs_hz": EXPECTED_FS_HZ,
                    "lead_index": 0,
                    "lead_name": "channel_0",
                    "start_sample_250hz": int(start),
                    "end_sample_250hz": int(start) + DEFAULT_CONTRACT.observation_samples,
                    "label": int(label),
                }
            )
    fields = list(rows[0])
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)
    return rows


def build(args: argparse.Namespace) -> dict:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    names = fetch_record_inventory(args.output_dir, args.base_url)
    pairs = parse_learning_pairs(names)
    if args.limit_pairs is not None:
        pairs = pairs[: args.limit_pairs]
    started = time.monotonic()
    summaries = []
    for index, pair in enumerate(pairs, start=1):
        metadata = process_pair(
            pair,
            args.output_dir,
            args.stage_dir,
            args.base_url,
            args.resume,
        )
        summaries.append(metadata)
        elapsed = time.monotonic() - started
        eta = elapsed / index * (len(pairs) - index)
        print(
            f"AFPDB pair {index}/{len(pairs)} {pair.subject_id}; "
            f"elapsed={elapsed / 60:.1f} min; ETA={eta / 60:.1f} min",
            flush=True,
        )
        atomic_json(
            args.output_dir / "status.json",
            {
                "phase": "preprocessing",
                "pairs_completed": index,
                "pairs_total": len(pairs),
                "elapsed_seconds": elapsed,
                "eta_seconds": eta,
            },
        )

    rows = _write_manifest(args.output_dir / "windows.csv", pairs, args.output_dir)
    labels = [int(row["label"]) for row in rows]
    if any(summary["window_count"] != 4 for summary in summaries):
        raise RuntimeError("every AFPDB subject must contribute exactly four windows")
    if any(summary["positive_count"] != 2 for summary in summaries):
        raise RuntimeError("every AFPDB subject must contribute two positive windows")
    summary = {
        "format": CACHE_FORMAT,
        "all_pairs_complete": True,
        "source": args.base_url,
        "license": "Open Data Commons Attribution License v1.0",
        "record_pair_groups": len(pairs),
        "identity_caveat": (
            "odd/even records are guaranteed to be one subject; PhysioNet does not "
            "identify which two of all 50 learning record sets repeat people"
        ),
        "records": 2 * len(pairs),
        "windows": len(rows),
        "class_counts": [labels.count(0), labels.count(1)],
        "window_policy": "two non-overlapping 10-minute observations per record",
        "positive_policy": "even p-record immediately preceding PAF",
        "negative_policy": "matched odd p-record distant from PAF",
        "manifest": "windows.csv",
        "manifest_sha256": _sha256(args.output_dir / "windows.csv"),
    }
    atomic_json(args.output_dir / "summary.json", summary)
    (args.output_dir / "COMPLETE").write_text("ok\n", encoding="utf-8")
    atomic_json(args.output_dir / "status.json", {"phase": "complete", **summary})
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--stage-dir", type=Path, default=Path("/content/afpdb_stage"))
    parser.add_argument("--base-url", default=AFPDB_BASE_URL)
    parser.add_argument("--limit-pairs", type=int, help="SMOKE TEST ONLY")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.limit_pairs is not None and args.limit_pairs < 1:
        parser.error("limit-pairs must be positive")
    build(args)


if __name__ == "__main__":
    main()

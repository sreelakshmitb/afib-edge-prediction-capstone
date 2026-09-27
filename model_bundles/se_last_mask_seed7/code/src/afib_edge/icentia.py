"""Build a bounded, patient-disjoint Icentia11k morphology-pretraining cache.

The full Icentia11k release is too large for a Colab workflow.  This module
downloads metadata first, deterministically selects patients that contain both
normal and AF/AFL rhythm intervals, then downloads only signal records needed
for a bounded set of 30-second examples.  Raw signal files are staged in a
temporary directory and are not retained.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import shutil
import time
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from .signal import bandpass_filter


ICENTIA_BASE_URL = (
    "https://physionet.org/files/icentia11k-continuous-ecg/1.0"
)
CACHE_FORMAT = "icentia_encoder_pretraining_cache_v1"
RHYTHM_TO_CLASS = {"N": 0, "AFIB": 1, "AFL": 1}
CLASS_NAMES = ("normal", "af_or_flutter")
SEGMENT_SECONDS = 30
EXPECTED_FS_HZ = 250


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_seed(text: str, seed: int) -> int:
    payload = f"{seed}:{text}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def download(url: str, destination: Path, retries: int = 5) -> Path:
    """Download one file atomically with bounded retries."""
    if destination.is_file() and destination.stat().st_size > 0:
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "afib-capstone/1.0"})
    for attempt in range(1, retries + 1):
        try:
            print(f"DOWNLOAD {url} (attempt {attempt}/{retries})", flush=True)
            with urllib.request.urlopen(request, timeout=180) as response:
                with temporary.open("wb") as handle:
                    shutil.copyfileobj(response, handle, length=1024 * 1024)
            if temporary.stat().st_size == 0:
                raise IOError("downloaded an empty file")
            temporary.replace(destination)
            return destination
        except urllib.error.HTTPError as error:
            if temporary.exists():
                temporary.unlink()
            if error.code == 404:
                raise FileNotFoundError(f"remote file does not exist: {url}") from error
            if attempt == retries:
                raise RuntimeError(f"download failed after {retries} attempts: {url}") from error
            time.sleep(min(2**attempt, 30))
        except (OSError, urllib.error.URLError) as error:
            if temporary.exists():
                temporary.unlink()
            if attempt == retries:
                raise RuntimeError(f"download failed after {retries} attempts: {url}") from error
            time.sleep(min(2**attempt, 30))
    raise AssertionError("unreachable")


def record_url(base_url: str, record: str, extension: str) -> str:
    return f"{base_url.rstrip('/')}/{record}.{extension}"


def patient_id(record: str) -> str:
    """Return ``pNNNNN`` from a patient directory or segment record path."""
    parts = Path(record.rstrip("/")).parts
    for part in reversed(parts):
        match = re.fullmatch(r"(p\d{5})(?:_s\d{2})?", part)
        if match:
            return match.group(1)
    raise ValueError(f"unexpected Icentia path: {record}")


def segment_records(patient_directory: str, limit: int) -> list[str]:
    """Expand one RECORDS patient directory into its possible segment names."""
    patient = patient_id(patient_directory)
    base = patient_directory.rstrip("/")
    return [f"{base}/{patient}_s{index:02d}" for index in range(limit)]


def local_metadata_base(root: Path, record: str) -> Path:
    return root / "metadata" / record


def fetch_records(output_dir: Path, base_url: str) -> tuple[str, ...]:
    path = download(f"{base_url.rstrip('/')}/RECORDS", output_dir / "RECORDS")
    records = tuple(
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    )
    if not records:
        raise RuntimeError("Icentia RECORDS is empty")
    if len(records) != len(set(records)):
        raise RuntimeError("Icentia RECORDS contains duplicate entries")
    return records


def fetch_metadata(output_dir: Path, base_url: str, record: str) -> Path:
    base = local_metadata_base(output_dir, record)
    download(record_url(base_url, record, "hea"), base.with_suffix(".hea"))
    download(record_url(base_url, record, "atr"), base.with_suffix(".atr"))
    return base


def parse_rhythm_intervals(record_base: Path) -> tuple[float, int, tuple[tuple[int, int, str], ...]]:
    """Parse Icentia's explicit ``(LABEL ... )`` rhythm regions."""
    try:
        import wfdb
    except ImportError as error:
        raise RuntimeError("Install wfdb before building the Icentia cache") from error

    header = wfdb.rdheader(str(record_base))
    annotation = wfdb.rdann(str(record_base), "atr")
    fs = float(header.fs)
    signal_samples = int(header.sig_len)
    if abs(fs - EXPECTED_FS_HZ) > 1e-6:
        raise ValueError(f"expected Icentia at 250 Hz, found {fs}")

    intervals: list[tuple[int, int, str]] = []
    active_start: int | None = None
    active_label: str | None = None
    for sample, raw_note in zip(annotation.sample, annotation.aux_note, strict=True):
        note = raw_note.replace("\x00", "").strip().upper()
        sample = int(sample)
        if note.startswith("("):
            token = note[1:].split()[0].strip()
            if active_start is not None and active_label is not None and sample > active_start:
                intervals.append((active_start, sample, active_label))
            active_start = sample
            active_label = token
        elif note.startswith(")") and active_start is not None and active_label is not None:
            if sample > active_start:
                intervals.append((active_start, sample, active_label))
            active_start = None
            active_label = None
    if active_start is not None and active_label is not None and signal_samples > active_start:
        intervals.append((active_start, signal_samples, active_label))
    return fs, signal_samples, tuple(intervals)


def choose_record_windows(
    record: str,
    intervals: tuple[tuple[int, int, str], ...],
    fs_hz: float,
    per_class: int,
    seed: int,
) -> list[dict[str, int]]:
    """Choose separated 30-second windows fully contained in rhythm regions."""
    length = int(round(SEGMENT_SECONDS * fs_hz))
    candidates: dict[int, list[int]] = defaultdict(list)
    for start, end, label in intervals:
        if label not in RHYTHM_TO_CLASS or end - start < length:
            continue
        class_index = RHYTHM_TO_CLASS[label]
        available = end - start - length
        if available == 0:
            starts = [start]
        else:
            count = min(per_class * 3, max(1, available // length + 1))
            starts = np.linspace(start, end - length, num=count, dtype=np.int64).tolist()
        candidates[class_index].extend(map(int, starts))

    selected: list[dict[str, int]] = []
    for class_index in range(len(CLASS_NAMES)):
        unique = sorted(set(candidates.get(class_index, [])))
        rng = random.Random(stable_seed(f"{record}:{class_index}", seed))
        rng.shuffle(unique)
        for start in sorted(unique[:per_class]):
            selected.append({"start": int(start), "label": int(class_index)})
    return sorted(selected, key=lambda row: (row["start"], row["label"]))


def select_examples(
    output_dir: Path,
    base_url: str,
    target_patients: int,
    max_scan_patients: int,
    metadata_records_per_patient: int,
    windows_per_class_per_record: int,
    max_windows_per_class_per_patient: int,
    seed: int,
) -> dict:
    selection_path = output_dir / "selection.json"
    config = {
        "base_url": base_url,
        "target_patients": target_patients,
        "max_scan_patients": max_scan_patients,
        "metadata_records_per_patient": metadata_records_per_patient,
        "windows_per_class_per_record": windows_per_class_per_record,
        "max_windows_per_class_per_patient": max_windows_per_class_per_patient,
        "seed": seed,
    }
    if selection_path.is_file():
        existing = json.loads(selection_path.read_text(encoding="utf-8"))
        if existing.get("config") != config:
            raise ValueError("selection configuration changed; use a new output directory")
        print("RESUME: using verified existing selection.json", flush=True)
        return existing

    # PhysioNet's root RECORDS file lists patient directories (11,000 rows),
    # not individual segment records. Each patient directory contains up to 50
    # records named pNNNNN_s00 ... pNNNNN_s49.
    patient_directories = list(fetch_records(output_dir, base_url))
    patients = [(patient_id(directory), directory) for directory in patient_directories]
    if len({patient for patient, _ in patients}) != len(patients):
        raise RuntimeError("Icentia RECORDS contains duplicate patient identifiers")
    random.Random(seed).shuffle(patients)

    selected_patients: list[str] = []
    selected_rows: list[dict] = []
    failures: list[dict[str, str]] = []
    scan_root = output_dir / "patient_scans"
    scan_root.mkdir(parents=True, exist_ok=True)
    scan_config_path = scan_root / "config.json"
    if scan_config_path.is_file():
        if json.loads(scan_config_path.read_text(encoding="utf-8")) != config:
            raise ValueError("metadata scan configuration changed; use a new output directory")
    else:
        atomic_json(scan_config_path, config)
    started = time.monotonic()
    for scan_index, (patient, patient_directory) in enumerate(
        patients[:max_scan_patients], start=1
    ):
        scan_path = scan_root / f"{patient}.json"
        if scan_path.is_file():
            scanned = json.loads(scan_path.read_text(encoding="utf-8"))
            if scanned.get("patient_id") != patient:
                raise ValueError(f"invalid saved patient scan: {scan_path}")
            patient_rows = scanned["records"]
            patient_failures = scanned["failures"]
            print(f"RESUME metadata scan: {patient}", flush=True)
        else:
            patient_rows: list[dict] = []
            patient_failures: list[dict[str, str]] = []
            patient_class_counts: dict[int, int] = defaultdict(int)
            patient_records = segment_records(
                patient_directory,
                metadata_records_per_patient,
            )
            # Raw metadata is staged locally, summarized to one small JSON per
            # patient, and discarded. This avoids thousands of tiny Drive files.
            with TemporaryDirectory(prefix="icentia_metadata_") as temporary_directory:
                metadata_stage = Path(temporary_directory)
                for record in patient_records:
                    try:
                        local = fetch_metadata(metadata_stage, base_url, record)
                        fs, signal_samples, intervals = parse_rhythm_intervals(local)
                        samples = choose_record_windows(
                            record,
                            intervals,
                            fs,
                            windows_per_class_per_record,
                            seed,
                        )
                        if samples:
                            patient_rows.append(
                                {
                                    "record": record,
                                    "patient_id": patient,
                                    "fs_hz": fs,
                                    "signal_samples": signal_samples,
                                    "samples": samples,
                                }
                            )
                            for sample in samples:
                                patient_class_counts[int(sample["label"])] += 1
                            if all(
                                patient_class_counts[class_index]
                                >= max_windows_per_class_per_patient
                                for class_index in range(len(CLASS_NAMES))
                            ):
                                break
                    except Exception as error:  # preserve occasional bad-record evidence
                        patient_failures.append({"record": record, "reason": repr(error)})
            atomic_json(
                scan_path,
                {
                    "patient_id": patient,
                    "records": patient_rows,
                    "failures": patient_failures,
                },
            )
        failures.extend(patient_failures)

        by_class: dict[int, list[tuple[int, int]]] = defaultdict(list)
        for record_index, row in enumerate(patient_rows):
            for sample_index, sample in enumerate(row["samples"]):
                by_class[int(sample["label"])].append((record_index, sample_index))
        if all(by_class.get(class_index) for class_index in range(len(CLASS_NAMES))):
            allowed: set[tuple[int, int]] = set()
            for class_index in range(len(CLASS_NAMES)):
                choices = by_class[class_index]
                rng = random.Random(stable_seed(f"{patient}:{class_index}", seed))
                rng.shuffle(choices)
                allowed.update(choices[:max_windows_per_class_per_patient])
            filtered_rows = []
            for record_index, row in enumerate(patient_rows):
                samples = [
                    sample
                    for sample_index, sample in enumerate(row["samples"])
                    if (record_index, sample_index) in allowed
                ]
                if samples:
                    filtered_rows.append({**row, "samples": samples})
            selected_patients.append(patient)
            selected_rows.extend(filtered_rows)

        elapsed = max(time.monotonic() - started, 1e-9)
        remaining = max(0, target_patients - len(selected_patients))
        eligible_rate = len(selected_patients) / elapsed
        eta = remaining / eligible_rate if eligible_rate > 0 else None
        eta_text = f"{eta / 60:.1f} min" if eta is not None else "estimating"
        print(
            f"SCAN patient {scan_index}/{min(len(patients), max_scan_patients)}; "
            f"eligible={len(selected_patients)}/{target_patients}; "
            f"elapsed={elapsed / 60:.1f} min; rough_ETA={eta_text}",
            flush=True,
        )
        atomic_json(
            output_dir / "status.json",
            {
                "phase": "metadata_scan",
                "patients_scanned": scan_index,
                "eligible_patients": len(selected_patients),
                "target_patients": target_patients,
                "elapsed_seconds": elapsed,
            },
        )
        if len(selected_patients) >= target_patients:
            break

    if len(selected_patients) < target_patients:
        raise RuntimeError(
            f"found only {len(selected_patients)} eligible patients; increase --max-scan-patients"
        )
    payload = {
        "format": "icentia_selection_v1",
        "config": config,
        "selected_patients": selected_patients,
        "records": selected_rows,
        "failed_metadata_records": failures,
    }
    atomic_json(selection_path, payload)
    return payload


def extract_chunks(output_dir: Path, base_url: str, selection: dict) -> list[dict]:
    chunks = output_dir / "chunks"
    chunks.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []
    failures: list[dict[str, str]] = []
    records = selection["records"]
    started = time.monotonic()
    for index, row in enumerate(records, start=1):
        safe_name = row["record"].replace("/", "__") + ".npz"
        destination = chunks / safe_name
        if destination.is_file():
            try:
                with np.load(destination, allow_pickle=False) as saved:
                    required = {"ecg", "labels", "starts", "patient_ids", "records"}
                    if not required.issubset(saved.files):
                        raise ValueError("missing arrays")
                print(f"RESUME chunk {index}/{len(records)}: {safe_name}", flush=True)
                results.append({"record": row["record"], "chunk": safe_name})
                continue
            except (OSError, ValueError, EOFError) as error:
                preserved = destination.with_name(
                    f"{destination.name}.incomplete-{time.time_ns()}"
                )
                destination.replace(preserved)
                print(
                    f"PRESERVED invalid chunk as {preserved.name}: {error}",
                    flush=True,
                )
        try:
            with TemporaryDirectory(prefix="icentia_signal_") as temporary_directory:
                stage = Path(temporary_directory)
                basename = Path(row["record"]).name
                base = stage / basename
                download(record_url(base_url, row["record"], "hea"), base.with_suffix(".hea"))
                download(record_url(base_url, row["record"], "dat"), base.with_suffix(".dat"))
                try:
                    import wfdb
                except ImportError as error:
                    raise RuntimeError("Install wfdb before extracting Icentia") from error
                record = wfdb.rdrecord(str(base), channels=[0], physical=True)
                signal = np.asarray(record.p_signal[:, 0], dtype=np.float64)
                if not np.isfinite(signal).all():
                    raise ValueError("non-finite signal")
                if abs(float(record.fs) - EXPECTED_FS_HZ) > 1e-6:
                    raise ValueError(f"unexpected sampling frequency {record.fs}")
                filtered = bandpass_filter(signal, float(record.fs))
                length = int(round(SEGMENT_SECONDS * float(record.fs)))
                examples, labels, starts = [], [], []
                for sample in row["samples"]:
                    start = int(sample["start"])
                    segment = filtered[start : start + length]
                    if segment.shape != (length,):
                        raise ValueError("selected segment is outside signal bounds")
                    examples.append(segment.astype(np.float16))
                    labels.append(int(sample["label"]))
                    starts.append(start)
                temporary = destination.with_suffix(destination.suffix + ".part")
                with temporary.open("wb") as handle:
                    np.savez_compressed(
                        handle,
                        ecg=np.stack(examples),
                        labels=np.asarray(labels, dtype=np.uint8),
                        starts=np.asarray(starts, dtype=np.int64),
                        patient_ids=np.asarray([row["patient_id"]] * len(labels)),
                        records=np.asarray([row["record"]] * len(labels)),
                    )
                temporary.replace(destination)
            results.append({"record": row["record"], "chunk": safe_name})
        except Exception as error:
            failures.append({"record": row["record"], "reason": repr(error)})
            print(f"EXTRACT FAILURE {row['record']}: {error}", flush=True)

        elapsed = time.monotonic() - started
        completed = index
        eta = elapsed / max(completed, 1) * (len(records) - completed)
        print(
            f"EXTRACT record {index}/{len(records)}; successes={len(results)}; "
            f"failures={len(failures)}; elapsed={elapsed / 60:.1f} min; "
            f"ETA={eta / 60:.1f} min",
            flush=True,
        )
        atomic_json(
            output_dir / "status.json",
            {
                "phase": "signal_extraction",
                "records_completed": completed,
                "records_total": len(records),
                "successes": len(results),
                "failures": len(failures),
                "elapsed_seconds": elapsed,
            },
        )
    atomic_json(output_dir / "extraction.json", {"chunks": results, "failures": failures})
    if not results:
        raise RuntimeError("no Icentia signal chunks were extracted")
    return results


def consolidate(output_dir: Path, chunks: list[dict], config: dict) -> dict:
    complete = output_dir / "COMPLETE"
    metadata_path = output_dir / "metadata.json"
    if complete.is_file() and metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        for filename, checksum in metadata["sha256"].items():
            path = output_dir / filename
            if not path.is_file() or digest_file(path) != checksum:
                raise ValueError(f"completed cache checksum mismatch: {filename}")
        print("RESUME: completed Icentia cache verified", flush=True)
        return metadata

    chunk_paths = [output_dir / "chunks" / item["chunk"] for item in chunks]
    counts = []
    for path in chunk_paths:
        with np.load(path, allow_pickle=False) as saved:
            counts.append(int(len(saved["labels"])))
    total = sum(counts)
    if total == 0:
        raise RuntimeError("Icentia chunks contain no samples")

    ecg_tmp = output_dir / "ecg.tmp.npy"
    ecg = np.lib.format.open_memmap(
        ecg_tmp,
        mode="w+",
        dtype=np.float16,
        shape=(total, EXPECTED_FS_HZ * SEGMENT_SECONDS),
    )
    labels = np.empty(total, dtype=np.uint8)
    patients = np.empty(total, dtype="U16")
    records = np.empty(total, dtype="U64")
    starts = np.empty(total, dtype=np.int64)
    position = 0
    for path, count in zip(chunk_paths, counts, strict=True):
        with np.load(path, allow_pickle=False) as saved:
            stop = position + count
            ecg[position:stop] = saved["ecg"]
            labels[position:stop] = saved["labels"]
            patients[position:stop] = saved["patient_ids"]
            records[position:stop] = saved["records"]
            starts[position:stop] = saved["starts"]
            position = stop
    ecg.flush()
    del ecg
    ecg_tmp.replace(output_dir / "ecg.npy")
    for name, array in (
        ("labels", labels),
        ("patient_ids", patients),
        ("records", records),
        ("starts", starts),
    ):
        temporary = output_dir / f"{name}.tmp.npy"
        np.save(temporary, array)
        temporary.replace(output_dir / f"{name}.npy")

    unique_patients = np.unique(patients)
    if len(unique_patients) < 5:
        raise RuntimeError("at least five patients are required for useful pretraining")
    for patient in unique_patients:
        if len(np.unique(labels[patients == patient])) != 2:
            raise RuntimeError(f"patient lost a class during extraction: {patient}")
    class_counts = np.bincount(labels, minlength=len(CLASS_NAMES)).astype(int).tolist()
    metadata = {
        "format": CACHE_FORMAT,
        "license": "CC BY-NC-SA 4.0; academic/non-commercial use",
        "source": config["base_url"],
        "samples": total,
        "patients": int(len(unique_patients)),
        "class_names": list(CLASS_NAMES),
        "class_counts": class_counts,
        "ecg_shape": [total, EXPECTED_FS_HZ * SEGMENT_SECONDS],
        "ecg_dtype": "float16",
        "normalization": "per-segment z-score applied lazily during training",
        "selection_config": config,
    }
    metadata["sha256"] = {
        filename: digest_file(output_dir / filename)
        for filename in ("ecg.npy", "labels.npy", "patient_ids.npy", "records.npy", "starts.npy")
    }
    atomic_json(metadata_path, metadata)
    complete.write_text("ok\n", encoding="utf-8")
    atomic_json(output_dir / "status.json", {"phase": "complete", **metadata})
    return metadata


def build(args: argparse.Namespace) -> dict:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selection = select_examples(
        args.output_dir,
        args.base_url,
        args.target_patients,
        args.max_scan_patients,
        args.metadata_records_per_patient,
        args.windows_per_class_per_record,
        args.max_windows_per_class_per_patient,
        args.seed,
    )
    chunks = extract_chunks(args.output_dir, args.base_url, selection)
    metadata = consolidate(args.output_dir, chunks, selection["config"])
    print(json.dumps(metadata, indent=2), flush=True)
    return metadata


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--base-url", default=ICENTIA_BASE_URL)
    parser.add_argument("--target-patients", type=int, default=100)
    parser.add_argument("--max-scan-patients", type=int, default=500)
    parser.add_argument("--metadata-records-per-patient", type=int, default=50)
    parser.add_argument("--windows-per-class-per-record", type=int, default=4)
    parser.add_argument("--max-windows-per-class-per-patient", type=int, default=40)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args(argv)
    positive = (
        args.target_patients,
        args.max_scan_patients,
        args.metadata_records_per_patient,
        args.windows_per_class_per_record,
        args.max_windows_per_class_per_patient,
    )
    if any(value < 1 for value in positive):
        parser.error("patient, record, and window limits must be positive")
    if args.target_patients > args.max_scan_patients:
        parser.error("target patients cannot exceed max scan patients")
    build(args)


if __name__ == "__main__":
    main()

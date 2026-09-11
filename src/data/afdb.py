"""MIT-BIH Atrial Fibrillation Database acquisition and WFDB loading."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from data.types import AFDBRecordMetadata, RhythmInterval

AFDB_DATABASE = "afdb"
AFDB_ANNOTATION_EXTENSION = "atr"
AF_RHYTHM_LABEL = "AFIB"


def download_afdb_record(
    raw_dir: Path,
    record_id: str,
    *,
    overwrite: bool = False,
) -> None:
    """Download one AFDB record and its rhythm annotations into an ignored local directory."""
    import wfdb

    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    wfdb.dl_database(
        AFDB_DATABASE,
        str(raw_dir),
        records=[record_id],
        annotators=[AFDB_ANNOTATION_EXTENSION],
        keep_subdirs=False,
        overwrite=overwrite,
    )


def _record_base(raw_dir: Path, record_id: str) -> str:
    return str(Path(raw_dir) / record_id)


def load_afdb_metadata(raw_dir: Path, record_id: str) -> AFDBRecordMetadata:
    """Read record metadata without loading the complete ECG signal."""
    import wfdb

    header = wfdb.rdheader(_record_base(raw_dir, record_id))
    if header.fs is None or header.sig_len is None:
        raise ValueError(f"Record {record_id} is missing fs or signal length metadata.")

    original_fs = float(header.fs)
    signal_length = int(header.sig_len)
    if original_fs <= 0 or signal_length <= 0:
        raise ValueError(f"Record {record_id} has invalid fs or signal length.")

    # REPRODUCTION CHOICE FOR THE SMOKE PIPELINE: retain record_id unchanged as
    # patient_id rather than inventing an unverified subject identifier. Before
    # GroupKFold is implemented, record-to-subject identity must be verified/mapped.
    patient_id = record_id
    signal_names = tuple(str(name) for name in (header.sig_name or ()))
    units = tuple(str(unit) for unit in (header.units or ()))

    return AFDBRecordMetadata(
        database=AFDB_DATABASE,
        record_id=record_id,
        patient_id=patient_id,
        original_fs_hz=original_fs,
        signal_length_samples=signal_length,
        duration_seconds=signal_length / original_fs,
        signal_names=signal_names,
        units=units,
    )


def normalize_rhythm_note(note: str | None) -> str | None:
    """Normalize WFDB auxiliary rhythm notes such as '(AFIB' and '(N'."""
    if note is None:
        return None
    text = str(note).replace("\x00", "").strip()
    if not text.startswith("("):
        return None
    label = text[1:].strip().upper()
    return label or None


def extract_rhythm_intervals(
    annotation_samples: Sequence[int],
    auxiliary_notes: Sequence[str | None],
    *,
    original_fs_hz: float,
    record_duration_seconds: float,
) -> tuple[RhythmInterval, ...]:
    """Convert rhythm-change annotations to time intervals on the original timeline."""
    if original_fs_hz <= 0:
        raise ValueError("original_fs_hz must be positive.")
    if record_duration_seconds <= 0:
        raise ValueError("record_duration_seconds must be positive.")
    if len(annotation_samples) != len(auxiliary_notes):
        raise ValueError("annotation_samples and auxiliary_notes must have equal length.")

    events: list[tuple[float, str]] = []
    for sample, note in zip(annotation_samples, auxiliary_notes, strict=True):
        rhythm = normalize_rhythm_note(note)
        if rhythm is None:
            continue
        time_seconds = float(sample) / original_fs_hz
        if time_seconds < 0:
            continue
        if time_seconds > record_duration_seconds:
            break
        events.append((time_seconds, rhythm))

    intervals: list[RhythmInterval] = []
    for index, (start_seconds, rhythm) in enumerate(events):
        end_seconds = (
            events[index + 1][0] if index + 1 < len(events) else record_duration_seconds
        )
        end_seconds = min(end_seconds, record_duration_seconds)
        if end_seconds > start_seconds:
            intervals.append(
                RhythmInterval(
                    start_seconds=start_seconds,
                    end_seconds=end_seconds,
                    rhythm=rhythm,
                )
            )
    return tuple(intervals)


def load_afdb_rhythm_intervals(
    raw_dir: Path,
    record_id: str,
    metadata: AFDBRecordMetadata | None = None,
) -> tuple[RhythmInterval, ...]:
    """Load AFDB rhythm-change annotations from the WFDB ``atr`` file."""
    import wfdb

    metadata = metadata or load_afdb_metadata(raw_dir, record_id)
    annotation = wfdb.rdann(_record_base(raw_dir, record_id), AFDB_ANNOTATION_EXTENSION)
    aux_notes = annotation.aux_note
    if aux_notes is None:
        raise ValueError(f"AFDB record {record_id} has no auxiliary rhythm annotations.")

    return extract_rhythm_intervals(
        annotation.sample,
        aux_notes,
        original_fs_hz=metadata.original_fs_hz,
        record_duration_seconds=metadata.duration_seconds,
    )


def extract_af_intervals(
    rhythm_intervals: Sequence[RhythmInterval],
) -> tuple[RhythmInterval, ...]:
    """Return only AFIB intervals from parsed rhythm intervals."""
    return tuple(interval for interval in rhythm_intervals if interval.is_af)


def load_afdb_signal_segment(
    raw_dir: Path,
    record_id: str,
    *,
    start_seconds: float,
    end_seconds: float,
    lead_index: int = 0,
    metadata: AFDBRecordMetadata | None = None,
) -> tuple[np.ndarray, float, str]:
    """Load one physical ECG lead for a bounded time interval from a local AFDB record."""
    import wfdb

    metadata = metadata or load_afdb_metadata(raw_dir, record_id)
    if start_seconds < 0 or end_seconds <= start_seconds:
        raise ValueError("Invalid signal interval.")
    if end_seconds > metadata.duration_seconds + 1e-9:
        raise ValueError("Requested signal interval exceeds record duration.")
    if lead_index < 0:
        raise ValueError("lead_index must be non-negative.")
    if metadata.signal_names and lead_index >= len(metadata.signal_names):
        raise ValueError(f"lead_index {lead_index} is out of range for record {record_id}.")

    sampfrom = int(round(start_seconds * metadata.original_fs_hz))
    sampto = int(round(end_seconds * metadata.original_fs_hz))
    record = wfdb.rdrecord(
        _record_base(raw_dir, record_id),
        sampfrom=sampfrom,
        sampto=sampto,
        channels=[lead_index],
        physical=True,
    )
    if record.p_signal is None or record.p_signal.ndim != 2 or record.p_signal.shape[1] != 1:
        raise ValueError(f"Record {record_id} did not return one physical ECG channel.")

    signal = np.asarray(record.p_signal[:, 0], dtype=np.float64)
    if not np.all(np.isfinite(signal)):
        raise ValueError(f"Record {record_id} contains non-finite ECG values in requested window.")
    lead_name = str(record.sig_name[0]) if record.sig_name else f"lead_{lead_index}"
    return signal, float(record.fs), lead_name

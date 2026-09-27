"""Rhythm annotation parsing independent of a particular WFDB database."""

from dataclasses import dataclass
from typing import Iterable


def normalize_rhythm_label(aux_note: str) -> str | None:
    """Return a normalized rhythm token, or None for a non-rhythm annotation."""
    note = aux_note.replace("\x00", "").strip()
    if not note.startswith("("):
        return None
    token = note[1:].split()[0].strip().upper()
    return token or None


def is_af_label(label: str) -> bool:
    return label in {"AF", "AFIB"}


@dataclass(frozen=True)
class RhythmInterval:
    start_sample: int
    end_sample: int
    label: str

    def __post_init__(self) -> None:
        if self.start_sample < 0 or self.end_sample <= self.start_sample:
            raise ValueError("rhythm interval must have positive duration")


def build_rhythm_intervals(
    annotation_samples: Iterable[int],
    aux_notes: Iterable[str],
    recording_samples: int,
) -> tuple[RhythmInterval, ...]:
    """Build a complete rhythm timeline.

    Any new rhythm annotation closes the previous rhythm. The final interval is
    closed using the actual ECG length, never the number of annotations.
    """
    if recording_samples <= 0:
        raise ValueError("recording_samples must be positive")

    events: list[tuple[int, str]] = []
    for sample, note in zip(annotation_samples, aux_notes, strict=True):
        label = normalize_rhythm_label(note)
        if label is None:
            continue
        sample = int(sample)
        if not 0 <= sample < recording_samples:
            raise ValueError(f"rhythm annotation sample {sample} is outside the recording")
        events.append((sample, label))

    events.sort(key=lambda item: item[0])
    intervals: list[RhythmInterval] = []
    for index, (start, label) in enumerate(events):
        end = events[index + 1][0] if index + 1 < len(events) else recording_samples
        if end > start:
            intervals.append(RhythmInterval(start, end, label))
    return tuple(intervals)


def af_intervals(rhythm_intervals: Iterable[RhythmInterval]) -> tuple[RhythmInterval, ...]:
    return tuple(interval for interval in rhythm_intervals if is_af_label(interval.label))


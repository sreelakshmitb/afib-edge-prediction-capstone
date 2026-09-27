"""Window metadata creation and leakage-resistant AF-onset labeling."""

from dataclasses import asdict, dataclass
from typing import Iterable

from .contracts import DEFAULT_CONTRACT, PipelineContract
from .rhythm import RhythmInterval, is_af_label


@dataclass(frozen=True)
class WindowRecord:
    dataset: str
    record_id: str
    subject_id: str
    start_sample: int
    end_sample: int
    horizon_end_sample: int
    label: int
    future_af_onset_sample: int | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class WindowAudit:
    accepted_negative: int = 0
    accepted_positive: int = 0
    excluded_af_overlap: int = 0
    excluded_non_sinus: int = 0
    excluded_incomplete_horizon: int = 0


def _overlaps(start: int, end: int, interval: RhythmInterval) -> bool:
    return start < interval.end_sample and end > interval.start_sample


def enumerate_windows(
    *,
    recording_samples: int,
    rhythm_intervals: Iterable[RhythmInterval],
    dataset: str,
    record_id: str,
    subject_id: str,
    contract: PipelineContract = DEFAULT_CONTRACT,
    require_annotated_sinus: bool = True,
    sinus_labels: frozenset[str] = frozenset({"N"}),
) -> tuple[tuple[WindowRecord, ...], WindowAudit]:
    """Create labels without materializing or copying ECG arrays.

    Windows lacking a complete future prediction horizon are excluded instead
    of being incorrectly labeled negative. When `require_annotated_sinus` is
    true, the complete observation must be covered by accepted sinus labels.
    """
    contract.validate()
    rhythms = tuple(sorted(rhythm_intervals, key=lambda item: item.start_sample))
    observation = contract.observation_samples
    horizon = contract.prediction_horizon_seconds * contract.target_fs_hz

    windows: list[WindowRecord] = []
    counts = {
        "accepted_negative": 0,
        "accepted_positive": 0,
        "excluded_af_overlap": 0,
        "excluded_non_sinus": 0,
        "excluded_incomplete_horizon": 0,
    }

    for start in range(0, max(0, recording_samples - observation + 1), contract.stride_samples):
        end = start + observation
        horizon_end = end + horizon
        if horizon_end > recording_samples:
            counts["excluded_incomplete_horizon"] += 1
            continue

        observation_rhythms = tuple(r for r in rhythms if _overlaps(start, end, r))
        if any(is_af_label(r.label) for r in observation_rhythms):
            counts["excluded_af_overlap"] += 1
            continue
        if require_annotated_sinus:
            covered = sum(
                max(0, min(end, r.end_sample) - max(start, r.start_sample))
                for r in observation_rhythms
                if r.label in sinus_labels
            )
            if covered != observation:
                counts["excluded_non_sinus"] += 1
                continue

        future_onsets = [
            r.start_sample
            for r in rhythms
            if is_af_label(r.label) and end <= r.start_sample < horizon_end
        ]
        onset = min(future_onsets) if future_onsets else None
        label = int(onset is not None)
        counts["accepted_positive" if label else "accepted_negative"] += 1
        windows.append(
            WindowRecord(
                dataset=dataset,
                record_id=record_id,
                subject_id=subject_id,
                start_sample=start,
                end_sample=end,
                horizon_end_sample=horizon_end,
                label=label,
                future_af_onset_sample=onset,
            )
        )
    return tuple(windows), WindowAudit(**counts)


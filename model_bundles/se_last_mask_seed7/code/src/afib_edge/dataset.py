"""Lazy record-wise access to the completed preprocessing cache."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from .contracts import DEFAULT_CONTRACT, HRV_FEATURE_NAMES
from .hrv_preprocess import HRVFoldPreprocessor


@dataclass(frozen=True)
class WindowRef:
    subject_id: str
    local_index: int
    label: int
    start_sample_250hz: int


def _safe_subject_name(subject_id: str) -> str:
    return subject_id.replace(":", "__")


def read_window_manifest(path: Path) -> dict[str, list[dict[str, str]]]:
    """Read and deterministically group the window manifest by record identity."""
    grouped: dict[str, list[dict[str, str]]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            grouped.setdefault(row["subject_id"], []).append(row)
    if not grouped:
        raise ValueError("window manifest is empty")
    for rows in grouped.values():
        rows.sort(key=lambda row: int(row["start_sample_250hz"]))
    return dict(sorted(grouped.items()))


class CacheWindowDataset(Dataset):
    """Dataset that memmaps one record at a time and returns fixed model inputs.

    The ECG cache is never assembled into one process-wide array. A sample is
    copied only when requested by ``__getitem__`` and returned as
    ``ecg=[20,1,7500]`` and ``hrv=[20,6]`` tensors.
    """

    def __init__(
        self,
        cache_root: Path,
        manifest_path: Path,
        hrv_preprocessor: HRVFoldPreprocessor | None = None,
        ecg_normalization: str = "none",
    ) -> None:
        super().__init__()
        self.cache_root = Path(cache_root)
        self.manifest_path = Path(manifest_path)
        self.hrv_preprocessor = hrv_preprocessor
        self.ecg_normalization = ecg_normalization.strip().lower().replace("-", "_")
        if self.ecg_normalization not in {"none", "segment_zscore"}:
            raise ValueError("ecg_normalization must be none or segment_zscore")
        grouped = read_window_manifest(self.manifest_path)
        self._refs: list[WindowRef] = []
        self._arrays: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}

        self._grouped = grouped
        for subject_id, rows in grouped.items():
            record_dir = self.cache_root / _safe_subject_name(subject_id)
            marker = record_dir / "COMPLETE"
            metadata_path = record_dir / "metadata.json"
            if not marker.is_file() or not metadata_path.is_file():
                raise FileNotFoundError(f"incomplete cache record: {record_dir}")
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if int(metadata["window_count"]) != len(rows):
                raise ValueError(f"manifest/cache window mismatch for {subject_id}")
            for local_index, row in enumerate(rows):
                self._refs.append(
                    WindowRef(
                        subject_id=subject_id,
                        local_index=local_index,
                        label=int(row["label"]),
                        start_sample_250hz=int(row["start_sample_250hz"]),
                    )
                )

        if not self._refs:
            raise ValueError("cache dataset is empty")

    @property
    def labels(self) -> np.ndarray:
        return np.asarray([ref.label for ref in self._refs], dtype=np.uint8)

    @property
    def subject_ids(self) -> np.ndarray:
        return np.asarray([ref.subject_id for ref in self._refs], dtype=object)

    @property
    def window_refs(self) -> tuple[WindowRef, ...]:
        return tuple(self._refs)

    def __len__(self) -> int:
        return len(self._refs)

    def __getitem__(self, index: int) -> dict[str, Tensor | str]:
        ref = self._refs[index]
        ecg_array, hrv_array, _, _ = self._get_arrays(ref.subject_id)
        ecg = np.array(ecg_array[ref.local_index], dtype=np.float32, copy=True)
        hrv = np.array(hrv_array[ref.local_index], dtype=np.float32, copy=True)
        if ecg.shape != (DEFAULT_CONTRACT.segments_per_window, DEFAULT_CONTRACT.segment_samples):
            raise ValueError(f"cached ECG violates the fixed shape for {ref.subject_id}")
        if hrv.shape != (DEFAULT_CONTRACT.segments_per_window, len(HRV_FEATURE_NAMES)):
            raise ValueError(f"cached HRV violates the fixed shape for {ref.subject_id}")
        if not np.isfinite(ecg).all():
            raise ValueError(f"non-finite ECG in {ref.subject_id}:{ref.local_index}")
        if self.ecg_normalization == "segment_zscore":
            # Scale each 30-second segment independently. This removes record
            # gain differences while retaining waveform morphology. The tiny
            # floor makes the behavior deterministic for near-flat segments.
            mean = ecg.mean(axis=-1, keepdims=True)
            scale = ecg.std(axis=-1, keepdims=True)
            ecg = (ecg - mean) / np.maximum(scale, 1e-6)
        if self.hrv_preprocessor is not None:
            hrv = self.hrv_preprocessor.transform(hrv)
        return {
            "ecg": torch.from_numpy(ecg[:, np.newaxis, :]),
            "hrv": torch.from_numpy(hrv),
            "label": torch.tensor(float(ref.label), dtype=torch.float32),
            "subject_id": ref.subject_id,
            "start_sample_250hz": torch.tensor(ref.start_sample_250hz, dtype=torch.int64),
        }

    def raw_hrv_for_indices(self, indices: Sequence[int]) -> np.ndarray:
        """Return raw HRV for fitting a fold preprocessor on selected windows."""
        selected = list(indices)
        result = np.empty(
            (
                len(selected),
                DEFAULT_CONTRACT.segments_per_window,
                len(HRV_FEATURE_NAMES),
            ),
            dtype=np.float32,
        )
        for output_index, dataset_index in enumerate(selected):
            ref = self._refs[int(dataset_index)]
            _, hrv_array, _, _ = self._get_arrays(ref.subject_id)
            result[output_index] = hrv_array[ref.local_index]
        return result

    def _get_arrays(
        self, subject_id: str
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if subject_id not in self._arrays:
            record_dir = self.cache_root / _safe_subject_name(subject_id)
            arrays = (
                np.load(record_dir / "ecg.npy", mmap_mode="r"),
                np.load(record_dir / "hrv.npy", mmap_mode="r"),
                np.load(record_dir / "labels.npy"),
                np.load(record_dir / "starts.npy"),
            )
            rows = self._grouped[subject_id]
            expected_labels = np.asarray([int(row["label"]) for row in rows])
            expected_starts = np.asarray([int(row["start_sample_250hz"]) for row in rows])
            if not np.array_equal(arrays[2], expected_labels):
                raise ValueError(f"cache labels disagree with manifest for {subject_id}")
            if not np.array_equal(arrays[3], expected_starts):
                raise ValueError(f"cache starts disagree with manifest for {subject_id}")
            if arrays[0].shape != (len(rows), 20, 7500) or arrays[1].shape != (len(rows), 20, 6):
                raise ValueError(f"cache array shape mismatch for {subject_id}")
            self._arrays[subject_id] = arrays
        return self._arrays[subject_id]


__all__ = ["CacheWindowDataset", "WindowRef", "read_window_manifest"]

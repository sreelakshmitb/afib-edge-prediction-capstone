"""Lazy access to the legacy and IRIDIA record-wise caches."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .contracts import DEFAULT_CONTRACT, HRV_FEATURE_NAMES
from .hrv_preprocess import HRVFoldPreprocessor


@dataclass(frozen=True)
class UnifiedRef:
    manifest_index: int
    cache_family: str
    cache_record_id: str
    cache_local_index: int
    subject_id: str
    source_family: str
    label: int


class UnifiedCacheDataset(Dataset):
    def __init__(
        self,
        legacy_cache: Path,
        iridia_cache: Path,
        manifest_path: Path,
        hrv_preprocessor: HRVFoldPreprocessor | None = None,
        ecg_normalization: str = "segment_zscore",
    ) -> None:
        self.roots = {"legacy": Path(legacy_cache), "iridia": Path(iridia_cache)}
        self.manifest = pd.read_csv(manifest_path).sort_values("manifest_index").reset_index(drop=True)
        if not np.array_equal(
            self.manifest.manifest_index.to_numpy(), np.arange(len(self.manifest))
        ):
            raise ValueError("manifest_index must be contiguous and equal dataset position")
        self.hrv_preprocessor = hrv_preprocessor
        self.ecg_normalization = ecg_normalization
        if ecg_normalization not in {"none", "segment_zscore"}:
            raise ValueError("unknown ECG normalization")
        self.refs = [
            UnifiedRef(
                manifest_index=int(row.manifest_index),
                cache_family=str(row.cache_family),
                cache_record_id=str(row.cache_record_id),
                cache_local_index=int(row.cache_local_index),
                subject_id=str(row.subject_id),
                source_family=str(row.source_family),
                label=int(row.label),
            )
            for row in self.manifest.itertuples(index=False)
        ]
        self._arrays: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
        for (family, record), rows in self.manifest.groupby(
            ["cache_family", "cache_record_id"], sort=True
        ):
            record_dir = self._record_dir(str(family), str(record))
            if not (record_dir / "COMPLETE").is_file():
                raise FileNotFoundError(f"incomplete cache record: {record_dir}")
            metadata = json.loads((record_dir / "metadata.json").read_text())
            expected = int(rows.cache_local_index.max()) + 1
            if int(metadata["window_count"]) != expected or len(rows) != expected:
                raise ValueError(f"cache/manifest count mismatch: {family}:{record}")
            if metadata["ecg_shape"] != [expected, 20, 7500]:
                raise ValueError(f"ECG metadata shape mismatch: {family}:{record}")
            if metadata["hrv_shape"] != [expected, 20, 6]:
                raise ValueError(f"HRV metadata shape mismatch: {family}:{record}")

    def _record_dir(self, family: str, record: str) -> Path:
        if family == "legacy":
            return self.roots[family] / record.replace(":", "__")
        if family == "iridia":
            return self.roots[family] / record
        raise ValueError(f"unknown cache family: {family}")

    def __len__(self) -> int:
        return len(self.refs)

    @property
    def labels(self) -> np.ndarray:
        return self.manifest.label.to_numpy(dtype=np.uint8)

    @property
    def subject_ids(self) -> np.ndarray:
        return self.manifest.subject_id.astype(str).to_numpy()

    def _get_arrays(self, ref: UnifiedRef) -> tuple[np.ndarray, np.ndarray]:
        key = (ref.cache_family, ref.cache_record_id)
        if key not in self._arrays:
            root = self._record_dir(*key)
            self._arrays[key] = (
                np.load(root / "ecg.npy", mmap_mode="r"),
                np.load(root / "hrv.npy", mmap_mode="r"),
            )
        return self._arrays[key]

    def raw_hrv_for_indices(self, indices: Sequence[int]) -> np.ndarray:
        output = np.empty((len(indices), 20, 6), dtype=np.float32)
        for destination, index in enumerate(indices):
            ref = self.refs[int(index)]
            _, hrv = self._get_arrays(ref)
            output[destination] = hrv[ref.cache_local_index]
        return output

    def __getitem__(self, index: int) -> dict:
        ref = self.refs[int(index)]
        manifest_row = self.manifest.iloc[int(index)]
        ecg_array, hrv_array = self._get_arrays(ref)
        ecg = np.array(ecg_array[ref.cache_local_index], dtype=np.float32, copy=True)
        hrv = np.array(hrv_array[ref.cache_local_index], dtype=np.float32, copy=True)
        if ecg.shape != (20, 7500) or hrv.shape != (20, 6):
            raise ValueError(f"tensor contract violation at manifest index {index}")
        if not np.isfinite(ecg).all():
            raise ValueError(f"non-finite ECG at manifest index {index}")
        if self.ecg_normalization == "segment_zscore":
            mean = ecg.mean(axis=-1, keepdims=True)
            scale = ecg.std(axis=-1, keepdims=True)
            ecg = (ecg - mean) / np.maximum(scale, 1e-6)
        if self.hrv_preprocessor is not None:
            hrv = self.hrv_preprocessor.transform(hrv)
        output = {
            "ecg": torch.from_numpy(ecg[:, None, :]),
            "hrv": torch.from_numpy(hrv),
            "label": torch.tensor(float(ref.label), dtype=torch.float32),
            "subject_id": ref.subject_id,
            "source_family": ref.source_family,
            "manifest_index": ref.manifest_index,
        }
        if "event_bin" in self.manifest.columns:
            output["event_bin"] = torch.tensor(
                int(manifest_row.event_bin), dtype=torch.long
            )
        return output

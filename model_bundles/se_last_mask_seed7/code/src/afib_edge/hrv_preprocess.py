"""Leakage-safe HRV imputation and standardization."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .contracts import HRV_FEATURE_NAMES


@dataclass
class HRVFoldPreprocessor:
    """Fit HRV statistics on training windows and reuse them unchanged.

    Missing values are imputed with training-fold medians. The imputed values
    are then standardized with training-fold means and standard deviations.
    No statistic is estimated from validation or outer-test windows.
    """

    epsilon: float = 1e-6
    median_: np.ndarray | None = None
    mean_: np.ndarray | None = None
    scale_: np.ndarray | None = None

    def fit(self, values: np.ndarray) -> "HRVFoldPreprocessor":
        x = self._validate(values)
        flat = x.reshape(-1, len(HRV_FEATURE_NAMES))
        finite = np.isfinite(flat)
        if not np.all(finite.any(axis=0)):
            missing = [
                HRV_FEATURE_NAMES[index]
                for index, has_value in enumerate(finite.any(axis=0))
                if not has_value
            ]
            raise ValueError(f"training fold has no finite values for: {missing}")

        medians = np.empty(len(HRV_FEATURE_NAMES), dtype=np.float64)
        for feature in range(len(HRV_FEATURE_NAMES)):
            medians[feature] = np.median(flat[finite[:, feature], feature])
        imputed = np.where(finite, flat, medians)
        means = np.mean(imputed, axis=0)
        scales = np.std(imputed, axis=0, ddof=0)
        scales = np.where(scales < self.epsilon, 1.0, scales)

        self.median_ = medians.astype(np.float32)
        self.mean_ = means.astype(np.float32)
        self.scale_ = scales.astype(np.float32)
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        self._check_fitted()
        x = self._validate(values).astype(np.float32, copy=True)
        finite = np.isfinite(x)
        x = np.where(finite, x, self.median_)
        x = (x - self.mean_) / self.scale_
        return np.asarray(x, dtype=np.float32)

    def fit_transform(self, values: np.ndarray) -> np.ndarray:
        return self.fit(values).transform(values)

    def state_dict(self) -> dict[str, object]:
        self._check_fitted()
        return {
            "feature_names": list(HRV_FEATURE_NAMES),
            "epsilon": self.epsilon,
            "median": self.median_.tolist(),
            "mean": self.mean_.tolist(),
            "scale": self.scale_.tolist(),
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, object]) -> "HRVFoldPreprocessor":
        """Restore fold-fitted statistics without refitting on evaluation data."""
        if list(state.get("feature_names", [])) != list(HRV_FEATURE_NAMES):
            raise ValueError("checkpoint HRV feature order does not match the model contract")
        instance = cls(epsilon=float(state["epsilon"]))
        instance.median_ = np.asarray(state["median"], dtype=np.float32)
        instance.mean_ = np.asarray(state["mean"], dtype=np.float32)
        instance.scale_ = np.asarray(state["scale"], dtype=np.float32)
        expected = (len(HRV_FEATURE_NAMES),)
        arrays = (instance.median_, instance.mean_, instance.scale_)
        if any(array.shape != expected for array in arrays):
            raise ValueError("checkpoint HRV statistics have an invalid shape")
        if not all(np.all(np.isfinite(array)) for array in arrays):
            raise ValueError("checkpoint HRV statistics must be finite")
        if np.any(instance.scale_ <= 0):
            raise ValueError("checkpoint HRV scales must be positive")
        return instance

    def _check_fitted(self) -> None:
        if self.median_ is None or self.mean_ is None or self.scale_ is None:
            raise RuntimeError("fit the HRV preprocessor on training data first")

    @staticmethod
    def _validate(values: np.ndarray) -> np.ndarray:
        x = np.asarray(values)
        if x.ndim < 2 or x.shape[-1] != len(HRV_FEATURE_NAMES):
            raise ValueError(
                f"HRV values must have a final dimension of {len(HRV_FEATURE_NAMES)}"
            )
        if x.size == 0:
            raise ValueError("HRV values cannot be empty")
        return x


__all__ = ["HRVFoldPreprocessor"]

"""An explicit experimental HRV recipe, fitted on inner-training windows only."""

from __future__ import annotations

import numpy as np

from .contracts import HRV_FEATURE_NAMES


class RobustHRVPreprocessor:
    """Mask invalid values, winsorize, log1p, impute, and standardize.

    The six fixed HRV features are nonnegative. Negative or nonfinite values
    become missing. Per-feature 0.5/99.5 percentiles, log-space medians, means,
    and scales are learned ONLY from the supplied inner-training observations.
    This is a candidate preprocessing recipe, not an established improvement.
    """

    def _clean(self, values):
        x = np.asarray(values, dtype=np.float64)
        if x.ndim != 3 or x.shape[1:] != (20, 6) or len(x) == 0:
            raise ValueError("HRV must be a non-empty [N,20,6] array")
        return np.where(np.isfinite(x) & (x >= 0), x, np.nan)

    def fit(self, values):
        x = self._clean(values).reshape(-1, 6)
        if not np.isfinite(x).any(axis=0).all():
            raise ValueError("inner training has an entirely missing HRV feature")
        self.lower = np.nanquantile(x, 0.005, axis=0)
        self.upper = np.nanquantile(x, 0.995, axis=0)
        z = np.log1p(np.clip(x, self.lower, self.upper))
        self.median = np.nanmedian(z, axis=0)
        z = np.where(np.isfinite(z), z, self.median)
        self.mean = z.mean(axis=0)
        scale = z.std(axis=0)
        self.scale = np.where(scale < 1e-6, 1.0, scale)
        self.missing_fraction = np.mean(~np.isfinite(x), axis=0)
        return self

    def transform(self, values):
        if not hasattr(self, "scale"):
            raise RuntimeError("fit on inner-training windows first")
        z = np.log1p(np.clip(self._clean(values), self.lower, self.upper))
        z = np.where(np.isfinite(z), z, self.median)
        return ((z - self.mean) / self.scale).astype(np.float32)

    def state_dict(self):
        return {
            "kind": "winsor_log1p_standardize_v1",
            "feature_names": list(HRV_FEATURE_NAMES),
            "quantiles": [0.005, 0.995],
            **{
                name: getattr(self, name).tolist()
                for name in ("lower", "upper", "median", "mean", "scale", "missing_fraction")
            },
        }

    @classmethod
    def from_state_dict(cls, state):
        if state.get("kind") != "winsor_log1p_standardize_v1":
            raise ValueError("unrecognized HRV preprocessing recipe")
        if state.get("feature_names") != list(HRV_FEATURE_NAMES):
            raise ValueError("HRV feature order mismatch")
        instance = cls()
        for name in ("lower", "upper", "median", "mean", "scale", "missing_fraction"):
            value = np.asarray(state[name], dtype=np.float64)
            if value.shape != (6,) or not np.isfinite(value).all():
                raise ValueError("invalid HRV preprocessing state")
            setattr(instance, name, value)
        if (instance.lower < 0).any() or (instance.upper < instance.lower).any():
            raise ValueError("invalid HRV clipping bounds")
        if (instance.scale <= 0).any():
            raise ValueError("invalid HRV scales")
        return instance


def summarize_hrv(values):
    """36 features: mean/std/median/first-5 mean/last-5 mean/temporal trend.

    Inputs are the twenty PRE-ONSET observation segments, never the future
    horizon. This deterministic summarization learns no population statistics.
    """
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 3 or x.shape[1:] != (20, 6) or not np.isfinite(x).all():
        raise ValueError("expected finite [N,20,6] preprocessed HRV")
    t = np.linspace(-1, 1, 20)
    trend = np.einsum("ntf,t->nf", x, t) / np.sum(t * t)
    return np.concatenate(
        [
            x.mean(1),
            x.std(1),
            np.median(x, axis=1),
            x[:, :5].mean(1),
            x[:, -5:].mean(1),
            trend,
        ],
        axis=1,
    )

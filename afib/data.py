"""Patient splits, inner-training-only HRV transform, strict tensor loading."""
import numpy as np
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
import torch
from torch.utils.data import Dataset


def patient_splits(rows, seed=2026):
    groups = np.array([r['patient'] for r in rows])
    labels = np.array([r['label'] for r in rows])
    if len(set(groups)) < 5:
        raise ValueError('Need at least five patients for outer GroupKFold')
    for fold, (development, test) in enumerate(GroupKFold(n_splits=5).split(labels, labels, groups)):
        inner = GroupShuffleSplit(n_splits=1, test_size=.25, random_state=seed)
        train_local, val_local = next(inner.split(development, labels[development], groups[development]))
        train, val = development[train_local], development[val_local]
        partitions = [set(groups[idx]) for idx in (train, val, test)]
        assert not (partitions[0] & partitions[1] or partitions[0] & partitions[2] or partitions[1] & partitions[2])
        if len(set(labels[train])) < 2:
            raise ValueError(f'Fold {fold}: training contains only one class')
        yield fold, train, val, test


class HRVTransform:
    def fit(self, values):
        values = np.asarray(values).reshape(-1, 6)
        self.median = np.array([np.median(col[np.isfinite(col)]) if np.isfinite(col).any() else 0.
                                for col in values.T], dtype=np.float32)
        filled = np.where(np.isfinite(values), values, self.median)
        self.mean = filled.mean(axis=0).astype(np.float32)
        self.scale = np.maximum(filled.std(axis=0), 1e-6).astype(np.float32)
        return self

    def __call__(self, values):
        filled = np.where(np.isfinite(values), values, self.median)
        return np.clip((filled-self.mean)/self.scale, -10, 10).astype(np.float32)

    def state(self):
        return {k: getattr(self, k).tolist() for k in ('median', 'mean', 'scale')}

    @classmethod
    def from_state(cls, state):
        instance = cls()
        for key in ('median', 'mean', 'scale'):
            setattr(instance, key, np.asarray(state[key], dtype=np.float32))
        return instance


def load_arrays(row):
    with np.load(row['path'], allow_pickle=False) as arrays:
        ecg, hrv = arrays['ecg'], arrays['hrv']
    if ecg.shape != (20, 1, 7500) or hrv.shape != (20, 6):
        raise ValueError(f'Invalid tensor shape: {row["path"]}')
    if not np.all(np.isfinite(ecg)):
        raise ValueError('Nonfinite ECG')
    return ecg, hrv


class AFibDataset(Dataset):
    def __init__(self, rows, transform):
        self.rows, self.transform = rows, transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        ecg, hrv = load_arrays(row)
        return torch.from_numpy(ecg), torch.from_numpy(self.transform(hrv)), torch.tensor(float(row['label']))

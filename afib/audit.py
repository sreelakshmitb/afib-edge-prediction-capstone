"""Read-only independent verification of a generated real-record manifest."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
import wfdb

from afib import HRV_ORDER
from afib.data import load_arrays
from afib.preprocess import rhythm_intervals


def audit(root):
    rows = json.loads((root / 'manifest.json').read_text())
    keys, missing, values = set(), np.zeros(6, dtype=int), []
    metadata = {}
    for row in rows:
        record = row['record']
        if record not in metadata:
            header = wfdb.rdheader(str(root / 'raw' / record))
            ann = wfdb.rdann(str(root / 'raw' / record), 'atr')
            metadata[record] = (header, rhythm_intervals(ann, header.sig_len))
        header, intervals = metadata[record]
        key = (record, row['start_sample'])
        assert key not in keys, f'Duplicate observation {key}'
        keys.add(key)
        start, end = row['start_sample'], row['end_sample']
        assert row['patient'] == record
        assert end-start == round(600*header.fs)
        horizon = end + round(1200*header.fs)
        assert horizon < header.sig_len
        normal_coverage = sum(max(0, min(b, end)-max(a, start)) for a,b,r in intervals if r == '(N')
        assert normal_coverage == end-start, f'Non-normal observation {key}'
        af_starts = [a for a,b,r in intervals if r == '(AFIB']
        assert row['label'] == int(any(end <= onset <= horizon for onset in af_starts))
        ecg, hrv = load_arrays(row)
        assert ecg.dtype == np.float32 and hrv.dtype == np.float32
        assert abs(float(ecg.mean())) < 1e-4
        assert abs(float(ecg.std())-1) < 1e-4
        missing += np.sum(~np.isfinite(hrv), axis=0)
        values.append(hrv)
    if not rows:
        raise ValueError('Empty manifest')
    values = np.concatenate(values)
    report = dict(windows=len(rows), patients=len(metadata), labels=dict(Counter(r['label'] for r in rows)),
        per_patient={p:dict(Counter(r['label'] for r in rows if r['patient']==p)) for p in sorted(metadata)},
        hrv_missing_fraction=dict(zip(HRV_ORDER, (missing/(len(rows)*20)).tolist())),
        hrv_medians={name:float(np.median(col[np.isfinite(col)])) if np.isfinite(col).any() else None
                     for name,col in zip(HRV_ORDER, values.T)},
        ecg_shape=[len(rows),20,1,7500], hrv_shape=[len(rows),20,6])
    print(json.dumps(report, indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    audit(Path(parser.parse_args().root))

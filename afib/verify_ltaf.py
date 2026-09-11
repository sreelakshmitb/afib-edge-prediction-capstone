"""Audit real LTAFDB observations before enabling bulk cohort generation."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.signal import butter, resample_poly, sosfilt
import wfdb
from wfdb.processing import xqrs_detect

from afib.data import load_arrays
from afib.ltaf import digest, merged_intervals, onset_label


def verify(root, record):
    out = root / 'processed-onset-v2' / record
    rows = json.loads((out / 'manifest.json').read_text())
    report = json.loads((out / 'report.json').read_text())
    raw = root / 'raw' / record
    header = wfdb.rdheader(str(raw))
    annotation = wfdb.rdann(str(raw), 'atr')
    intervals = merged_intervals(annotation, header.sig_len)
    assert {r['label'] for r in rows} == {0, 1}, 'Smoke verification needs both classes'
    for row in rows:
        assert row['patient'] == 'ltafdb:'+record
        start, end = row['start_sample'], row['end_sample']
        horizon = end+int(header.fs*1200)
        assert end-start == header.fs*600
        assert row['label'] == onset_label(start,end,horizon,intervals,header.sig_len)
        # Independent state transition scan: only N -> AFIB or other -> AFIB is onset.
        actual_onsets = [a for j,(a,b,r) in enumerate(intervals)
                         if r == '(AFIB' and j and intervals[j-1][2] != '(AFIB']
        assert row['label'] == int(any(end <= a <= horizon for a in actual_onsets))
        ecg, hrv = load_arrays(row)
        assert np.isfinite(ecg).all() and np.isfinite(hrv[:,3]).sum() >= 10
        assert np.isnan(hrv[:9,2]).all()
    peak_checks = []
    # Annotation comparison is diagnostic only; annotations never supply model HRV.
    selected = [next(r for r in rows if r['label']==label) for label in (0,1)]
    for row in selected:
        start, end = row['start_sample'], row['end_sample']
        signal = wfdb.rdrecord(str(raw),sampfrom=start,sampto=end,channels=[0]).p_signal[:,0]
        assert len(signal)==int(600*header.fs)
        filtered = sosfilt(butter(4,[.5,40],fs=250,btype='bandpass',output='sos'),resample_poly(signal,125,64))
        detected = xqrs_detect(filtered,fs=250,verbose=False)/250
        beats = np.array([s for s,t in zip(annotation.sample,annotation.symbol)
                          if start<=s<end and t in ('N','V','A','Q','L','R','a','J','S','F','e','j','E','/','f')])
        expected = (beats-start)/header.fs
        i=j=matched=0
        while i<len(detected) and j<len(expected):
            delta=detected[i]-expected[j]
            if abs(delta)<=.1:
                matched+=1; i+=1; j+=1
            elif delta<0:
                i+=1
            else:
                j+=1
        peak_checks.append(dict(start_sample=start,label=row['label'],detected=len(detected),
            reviewed=len(expected),matched=matched,tolerance_seconds=.1,
            sensitivity=matched/len(expected),precision=matched/len(detected)))
        # Recomputed tensor uses exactly observation input; never reads future ECG.
        normalized=(filtered.astype(np.float32)-filtered.astype(np.float32).mean())/filtered.astype(np.float32).std()
        np.testing.assert_allclose(load_arrays(row)[0].reshape(-1),normalized,atol=1e-6)
    assert all(c['sensitivity'] >= .9 and c['precision'] >= .9 for c in peak_checks), 'Inspect poor R-peak agreement before bulk processing'
    result=dict(record=record,passed=True,windows=len(rows),labels=report['labels'],
        native_fs=header.fs,model_fs=250,ecg_shape=[len(rows),20,1,7500],hrv_shape=[len(rows),20,6],
        peak_checks=peak_checks,hrv_missing_fraction=report['hrv_missing_fraction'],
        source_sha256=report['source_sha256'],manifest_sha256=digest(out/'manifest.json'),
        identity_verification='Record ID and dataset namespace verified; biological cross-dataset linkage unavailable')
    (out/'verification.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',required=True)
    parser.add_argument('--record',default='00')
    args=parser.parse_args()
    verify(Path(args.root),args.record)

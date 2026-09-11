"""Versioned LTAFDB ingestion; leaves PILOT_V1 inputs and runs untouched."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path

import numpy as np
import requests
import wfdb

from afib import HRV_ORDER
from afib.preprocess import external_root, preprocess_observation, rhythm_intervals


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def merged_intervals(annotation, length):
    """Collapse repeated rhythm markers so they cannot create false onsets."""
    merged = []
    for a, b, rhythm in rhythm_intervals(annotation, length):
        if merged and merged[-1][2] == rhythm and merged[-1][1] == a:
            merged[-1] = (merged[-1][0], b, rhythm)
        else:
            merged.append((a, b, rhythm))
    return merged


def onset_label(start, end, horizon, intervals, length):
    """Known non-AF observation; any AF onset in fully annotated follow-up.

    Keep the initial N-only observation rule. Other known future rhythms do not
    erase an AF outcome. Unknown/gapped follow-up is censored for both classes.
    """
    if horizon >= length:
        return None
    normal = sum(max(0, min(b, end)-max(a, start)) for a,b,r in intervals if r == '(N')
    if normal != end-start:
        return None
    future = [(a,b,r) for a,b,r in intervals if a <= horizon and b > end]
    coverage = sum(max(0, min(b,horizon+1)-max(a,end)) for a,b,r in future)
    if coverage != horizon+1-end or any(r == 'UNKNOWN' for a,b,r in future):
        return None
    return int(any(end <= a <= horizon and r == '(AFIB' for a,b,r in future))


def fetch(name, raw, session):
    if Path(name).name != name:
        raise ValueError('Expected a filename')
    target = raw / name
    if target.exists():
        return target
    url = 'https://physionet-open.s3.amazonaws.com/ltafdb/1.0.0/' + name
    if name == 'SHA256SUMS':
        url = 'https://archive.physionet.org/physiobank/database/ltafdb/SHA256SUMS'
    partial = raw / (name + '.partial')
    with session.head(url, timeout=(15, 30)) as response:
        response.raise_for_status()
        size = int(response.headers['Content-Length'])
    offset = partial.stat().st_size if partial.exists() else 0
    if offset > size:
        raise ValueError('Oversized partial file')
    while offset < size:
        end = min(offset + 4*1024**2, size)-1
        for attempt in range(3):
            try:
                with session.get(url, headers={'Range': f'bytes={offset}-{end}'}, timeout=(15, 30)) as response:
                    response.raise_for_status()
                    if response.status_code != 206 or response.headers.get('Content-Range') != f'bytes {offset}-{end}/{size}':
                        raise ValueError('Invalid byte range response')
                    data = response.content
                    if len(data) != end-offset+1:
                        raise ValueError('Truncated download')
                break
            except requests.RequestException:
                if attempt == 2:
                    raise
        with partial.open('ab') as stream:
            stream.write(data)
        offset += len(data)
        print(f'{name}: {offset}/{size}', flush=True)
    partial.replace(target)
    return target


def process(record, root):
    raw = root / 'raw'
    raw.mkdir(parents=True, exist_ok=True)
    out = root / 'processed-onset-v2' / record
    if (out / 'report.json').exists():
        print(f'Preserving completed LTAFDB {record}', flush=True)
        return
    with requests.Session() as session:
        sums = fetch('SHA256SUMS', raw, session)
        checksums = {line.split()[1].lstrip('*'): line.split()[0] for line in sums.read_text().splitlines()}
        hashes = {}
        for ext in ('hea', 'atr'):
            name = f'{record}.{ext}'
            path = fetch(name, raw, session)
            hashes[name] = digest(path)
            if hashes[name] != checksums[name]:
                raise ValueError(f'Source checksum mismatch: {name}')
    header = wfdb.rdheader(str(raw / record))
    ann = wfdb.rdann(str(raw / record), 'atr')
    intervals = merged_intervals(ann, header.sig_len)
    candidates=[]
    for start in range(0, header.sig_len, round(header.fs*600)):
        end, horizon = start+round(header.fs*600), start+round(header.fs*1800)
        label=onset_label(start,end,horizon,intervals,header.sig_len)
        if label is not None:
            candidates.append((start,end,label))
    # Sustained AF and other ineligible records need annotation screening only.
    # Do not transfer a large signal file that cannot provide an observation.
    if candidates:
        name=f'{record}.dat'
        with requests.Session() as session:
            path=fetch(name,raw,session)
        hashes[name]=digest(path)
        if hashes[name]!=checksums[name]:
            raise ValueError(f'Source checksum mismatch: {name}')
    out.mkdir(parents=True, exist_ok=True)
    rows, rejected = [], Counter()
    rejected['rhythm_or_followup']=len(range(0,header.sig_len,round(header.fs*600)))-len(candidates)
    for start,end,label in candidates:
        horizon=end+round(header.fs*1200)
        # Only the observation is passed to signal processing; annotation follow-up is label-only.
        signal = wfdb.rdrecord(str(raw / record), sampfrom=start, sampto=end, channels=[0]).p_signal[:, 0]
        try:
            ecg, hrv, count = preprocess_observation(signal, header.fs)
        except ValueError as exc:
            rejected[str(exc)] += 1
            continue
        assert ecg.shape == (20, 1, 7500) and hrv.shape == (20, 6)
        coverage = sum(max(0, min(b, end)-max(a, start)) for a, b, r in intervals if r == '(N')
        assert coverage == end-start
        onset = [a for a, b, r in intervals if r == '(AFIB' and end <= a <= horizon]
        assert label == int(bool(onset))
        path = out / f'{start}.npz'
        np.savez_compressed(path, ecg=ecg, hrv=hrv)
        rows.append(dict(dataset='ltafdb', patient=f'ltafdb:{record}', record=record,
            start_sample=start, end_sample=end, native_fs=header.fs,
            start_seconds=start/header.fs, end_seconds=end/header.fs,
            label=label, path=str(path), r_peaks=count,
            hrv_missing=int(np.isnan(hrv).sum()), onset_seconds=[a/header.fs for a in onset]))
    report = dict(dataset='ltafdb', record=record, label_policy='onset-v2', native_fs=header.fs, channel=0,
        signal_download_required=bool(candidates),candidate_labels=dict(Counter(c[2] for c in candidates)),
        lead=header.sig_name[0], source_sha256=hashes, rhythm_intervals=intervals,
        accepted=len(rows), labels=dict(Counter(r['label'] for r in rows)), rejected=dict(rejected),
        hrv_order=HRV_ORDER, preprocessing_sha256=digest(Path(__file__).with_name('preprocess.py')),
        adapter_sha256=digest(__file__), annotation='atr: reviewed beat/rhythm annotations',
        identity='dataset-namespaced record; one record per subject assumption; no cross-dataset person identifier')
    if rows:
        features = []
        for row in rows:
            with np.load(row['path'], allow_pickle=False) as arrays:
                features.append(arrays['hrv'])
        hrv = np.stack(features)
        report['hrv_missing_fraction'] = dict(zip(HRV_ORDER, np.mean(~np.isfinite(hrv), axis=(0, 1)).tolist()))
    (out / 'manifest.json').write_text(json.dumps(rows, indent=2))
    # Completion marker written last. Interrupted records may resume, completed records never reprocess.
    completion=out/'report.json.partial'
    completion.write_text(json.dumps(report,indent=2))
    completion.replace(out/'report.json')
    print(json.dumps({k:v for k,v in report.items() if k not in ('rhythm_intervals','source_sha256')}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--records', nargs='+', required=True)
    parser.add_argument('--workers',type=int,choices=[1,2],default=1)
    args = parser.parse_args()
    root = external_root(args.root)
    records=args.records
    if records==['all']:
        verification=root/'processed-onset-v2/00/verification.json'
        if not verification.exists() or not json.loads(verification.read_text())['passed']:
            raise ValueError('One real-record verification must pass before bulk acquisition')
        records=sorted(p.split()[1][:-4] for p in (root/'raw/SHA256SUMS').read_text().splitlines()
                       if p.split()[1].endswith('.dat') and '/' not in p.split()[1])
        if len(records)!=84:
            raise ValueError('Unexpected LTAFDB inventory')
    for record in records:
        if not record.isdigit():
            raise ValueError('Numeric LTAFDB record identifier required')
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        list(executor.map(lambda record: process(record,root),records))


if __name__ == '__main__':
    main()

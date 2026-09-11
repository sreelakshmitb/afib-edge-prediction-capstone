"""AFDB preprocessing. No learned population statistics are computed here."""
import argparse
from collections import Counter
from fractions import Fraction
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.signal import butter, resample_poly, sosfilt, welch
import wfdb
from wfdb.processing import xqrs_detect

from afib import FS, OBS_SECONDS, HORIZON_SECONDS, HRV_ORDER


def external_root(value):
    root = Path(value).expanduser().resolve()
    repo = Path(__file__).resolve().parents[1]
    if root == repo or repo in root.parents:
        raise ValueError("Data/artifact root must be outside the repository")
    root.mkdir(parents=True, exist_ok=True)
    return root


def rhythm_intervals(annotation, length):
    changes = [(int(s), a.strip().rstrip('\x00')) for s, a in
               zip(annotation.sample, annotation.aux_note) if a.startswith('(')]
    intervals = []
    previous, rhythm = 0, "UNKNOWN"
    for sample, new_rhythm in changes:
        if sample > previous:
            intervals.append((previous, sample, rhythm))
        previous, rhythm = sample, new_rhythm
    if previous < length:
        intervals.append((previous, length, rhythm))
    return intervals


def label_window(start, end, horizon_end, intervals, length):
    """Observation [start,end); onset [end,horizon_end], complete follow-up."""
    if horizon_end >= length:
        return None
    observation = [(a, b, r) for a, b, r in intervals if a < end and b > start]
    if not observation or any(r != '(N' for _, _, r in observation):
        return None
    # Unknown rhythm, flutter and junctional rhythm are excluded from follow-up.
    future = [(a, b, r) for a, b, r in intervals if a <= horizon_end and b > end]
    if not future or any(r not in ('(N', '(AFIB') for _, _, r in future):
        return None
    return int(any(end <= a <= horizon_end and r == '(AFIB' for a, _, r in future))


def sample_entropy(rr):
    if len(rr) < 10 or np.std(rr) == 0:
        return np.nan
    tolerance = .2 * np.std(rr, ddof=1)
    # Equal template counts for m=2 and m+1; exclude self matches.
    n = len(rr) - 2
    counts = []
    for m in (2, 3):
        templates = np.lib.stride_tricks.sliding_window_view(rr, m)[:n]
        distances = np.max(np.abs(templates[:, None] - templates[None, :]), axis=2)
        counts.append(np.count_nonzero(np.triu(distances <= tolerance, 1)))
    return -np.log(counts[1] / counts[0]) if all(counts) else np.nan


def hrv_features(peaks, segment_end):
    """Time-domain seconds converted to ms; pNN50 percent; spectral ratio unitless."""
    times = np.asarray(peaks, dtype=float) / FS
    rr = np.diff(times)
    ends = times[1:]
    starts = times[:-1]
    good = (rr >= .3) & (rr <= 2.)
    local_mask = (starts >= segment_end - 30) & (ends < segment_end)
    local = rr[local_mask]
    result = np.full(6, np.nan, dtype=np.float32)
    # Do not join RR differences across a rejected beat interval.
    if len(local) >= 10 and np.all(good[local_mask]):
        delta = np.diff(local)
        result[[0, 1, 3, 4, 5]] = [np.sqrt(np.mean(delta**2))*1000,
            np.std(local, ddof=1)*1000, np.mean(local)*1000,
            np.mean(np.abs(delta) > .05)*100, sample_entropy(local)]
    # Require five minutes of causal context for LF/HF; no padding with future RR.
    spectral_mask = (starts >= segment_end - 300) & (ends < segment_end)
    r, t = rr[spectral_mask], ends[spectral_mask]
    if segment_end >= 300 and len(r) >= 100 and np.all(good[spectral_mask]) and t[-1]-t[0] >= 290:
        uniform = np.interp(np.arange(t[0], t[-1], .25), t, r)
        freq, power = welch(uniform, fs=4., nperseg=min(1024, len(uniform)))
        lf = (freq >= .04) & (freq < .15)
        hf = (freq >= .15) & (freq <= .4)
        high = np.trapezoid(power[hf], freq[hf])
        if high > 1e-12:
            result[2] = np.trapezoid(power[lf], freq[lf]) / high
    return result


def preprocess_observation(signal, native_fs):
    if not np.all(np.isfinite(signal)):
        raise ValueError("nonfinite_signal")
    # Reject >=1 second exact flat runs, including AFDB's missing tape blocks.
    equal = np.diff(signal) == 0
    edges = np.diff(np.r_[False, equal, False].astype(int))
    if np.any(np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1) >= native_fs):
        raise ValueError("flat_signal")
    if native_fs != FS:
        ratio = Fraction(FS / native_fs).limit_denominator(10000)
        signal = resample_poly(signal, ratio.numerator, ratio.denominator)
    if len(signal) != FS * OBS_SECONDS:
        raise ValueError("unexpected_length")
    # Causal filter; neither filtering nor detection sees prediction-horizon ECG.
    filtered = sosfilt(butter(4, [.5, 40], fs=FS, btype='bandpass', output='sos'), signal)
    peaks = xqrs_detect(filtered, fs=FS, verbose=False)
    if len(peaks) < 200:
        raise ValueError("insufficient_peaks")
    hrv = np.stack([hrv_features(peaks, end) for end in range(30, 601, 30)])
    if np.isfinite(hrv[:, 3]).sum() < 10:
        raise ValueError("insufficient_valid_rr_segments")
    ecg = filtered.reshape(20, 1, 7500).astype(np.float32)
    # Per-observation normalization uses only available ECG, no population fit.
    scale = float(ecg.std())
    if scale < 1e-6:
        raise ValueError("constant_signal")
    ecg = (ecg - ecg.mean()) / scale
    return ecg, hrv, len(peaks)


def process_record(record, root):
    raw, out = root / 'raw', root / 'processed' / record
    raw.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    required = [f'{record}.{ext}' for ext in ('hea', 'dat', 'atr')]
    missing = [f for f in required if not (raw / f).exists()]
    if missing:
        wfdb.dl_files('afdb', str(raw), missing)
    header = wfdb.rdheader(str(raw / record))
    ann = wfdb.rdann(str(raw / record), 'atr')
    intervals = rhythm_intervals(ann, header.sig_len)
    rows, rejected = [], Counter()
    for start in range(0, header.sig_len, round(header.fs * OBS_SECONDS)):
        end = start + round(header.fs * OBS_SECONDS)
        future = end + round(header.fs * HORIZON_SECONDS)
        label = label_window(start, end, future, intervals, header.sig_len)
        if label is None:
            rejected['rhythm_or_followup'] += 1
            continue
        path = out / f'{start}.npz'
        signal = wfdb.rdrecord(str(raw / record), sampfrom=start, sampto=end, channels=[0]).p_signal[:, 0]
        try:
            ecg, hrv, npeaks = preprocess_observation(signal, header.fs)
        except ValueError as exc:
            rejected[str(exc)] += 1
            continue
        np.savez_compressed(path, ecg=ecg, hrv=hrv)
        rows.append(dict(patient=record, record=record, start_sample=start,
            end_sample=end, native_fs=header.fs, start_seconds=start/header.fs,
            end_seconds=end/header.fs, label=label, path=str(path), r_peaks=npeaks,
            hrv_missing=int(np.isnan(hrv).sum())))
    hashes = {name: hashlib.sha256((raw / name).read_bytes()).hexdigest() for name in required}
    report = dict(record=record, native_fs=header.fs, channel=0, lead=header.sig_name[0],
        rhythm_intervals=intervals, accepted=len(rows), labels=dict(Counter(r['label'] for r in rows)),
        rejected=dict(rejected), source_sha256=hashes, hrv_order=HRV_ORDER)
    (out / 'manifest.json').write_text(json.dumps(rows, indent=2))
    (out / 'report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k not in ('rhythm_intervals', 'source_sha256') }), flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--records', nargs='+', default=['04015'])
    args = parser.parse_args()
    root = external_root(args.root)
    records = args.records
    if records == ['all']:
        records = [r for r in wfdb.get_record_list('afdb') if r not in ('00735', '03665')]
    rows = []
    for record in records:
        rows.extend(process_record(record, root))
    (root / 'manifest.json').write_text(json.dumps(rows, indent=2))
    print(f'Manifest: {root / "manifest.json"}; {len(rows)} windows', flush=True)


if __name__ == '__main__':
    main()

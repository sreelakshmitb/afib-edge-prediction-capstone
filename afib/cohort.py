"""Build a new combined manifest without changing PILOT_V1 files."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
import wfdb

from afib import HRV_ORDER
from afib.data import load_arrays
from afib.ltaf import digest, merged_intervals, onset_label
from afib.preprocess import external_root, preprocess_observation


def build(pilot, ltaf, out):
    if (out/'manifest.json').exists():
        raise FileExistsError('Preserve existing combined manifest; use another cohort version')
    reports=list((ltaf/'processed-onset-v2').glob('*/report.json'))
    if len(reports)!=84:
        raise ValueError(f'Expected 84 completed LTAFDB records, found {len(reports)}')
    rows=[]; source_hashes={}; excluded={}; label_changes=Counter()
    for dataset, root, report_paths in (
        ('afdb',pilot,sorted((pilot/'processed').glob('*/report.json'))),
        ('ltafdb',ltaf,sorted(reports))):
        if dataset=='afdb' and len(report_paths)!=23:
            raise ValueError('Incomplete PILOT_V1 source inventory')
        for report_path in report_paths:
            report=json.loads(report_path.read_text()); record=report['record']
            patient=f'{dataset}:{record}'
            for name,sha in report['source_sha256'].items():
                if digest(root/'raw'/name)!=sha:
                    raise ValueError(f'Changed source file: {name}')
                if name.endswith('.dat'):
                    if sha in source_hashes:
                        raise ValueError(f'Duplicate recording: {patient} and {source_hashes[sha]}')
                    source_hashes[sha]=patient
            old=json.loads(report_path.with_name('manifest.json').read_text())
            if dataset=='ltafdb':
                if report.get('label_policy')!='onset-v2':
                    raise ValueError('Wrong label version')
                rows.extend(old); excluded[patient]=report['rejected']
                continue
            # Reuse identical observation features. Only newly eligible AFDB windows
            # need signal processing; no PILOT_V1 arrays or labels are overwritten.
            lookup={r['start_sample']:r for r in old}
            header=wfdb.rdheader(str(root/'raw'/record))
            intervals=merged_intervals(wfdb.rdann(str(root/'raw'/record),'atr'),header.sig_len)
            rejected=Counter()
            for start in range(0,header.sig_len,round(header.fs*600)):
                end=start+round(header.fs*600)
                label=onset_label(start,end,end+round(header.fs*1200),intervals,header.sig_len)
                if label is None:
                    rejected['rhythm_or_followup']+=1; continue
                if start in lookup:
                    row=dict(lookup[start]); label_changes[f"reused_{row['label']}_to_{label}"]+=1
                    row.update(dataset=dataset,patient=patient,label=label)
                else:
                    signal=wfdb.rdrecord(str(root/'raw'/record),sampfrom=start,sampto=end,channels=[0]).p_signal[:,0]
                    try:
                        ecg,hrv,peaks=preprocess_observation(signal,header.fs)
                    except ValueError as exc:
                        rejected[str(exc)]+=1; continue
                    directory=out/'additional-afdb'/record; directory.mkdir(parents=True,exist_ok=True)
                    path=directory/f'{start}.npz'; np.savez_compressed(path,ecg=ecg,hrv=hrv)
                    row=dict(dataset=dataset,patient=patient,record=record,start_sample=start,
                        end_sample=end,native_fs=header.fs,start_seconds=start/header.fs,
                        end_seconds=end/header.fs,label=label,path=str(path),r_peaks=peaks,
                        hrv_missing=int(np.isnan(hrv).sum()))
                    label_changes[f'new_{label}']+=1
                rows.append(row)
            excluded[patient]=dict(rejected)
    rows.sort(key=lambda r:(r['patient'],r['start_sample']))
    identities=set(); missing=np.zeros(6); per_patient={}; metadata={}
    for row in rows:
        key=(row['patient'],row['start_sample']); assert key not in identities; identities.add(key)
        dataset=row['dataset']; root=pilot if dataset=='afdb' else ltaf
        if row['patient'] not in metadata:
            raw=root/'raw'/row['record']; h=wfdb.rdheader(str(raw))
            metadata[row['patient']]=(h,merged_intervals(wfdb.rdann(str(raw),'atr'),h.sig_len))
        h,intervals=metadata[row['patient']]
        assert row['label']==onset_label(row['start_sample'],row['end_sample'],row['end_sample']+round(h.fs*1200),intervals,h.sig_len)
        ecg,hrv=load_arrays(row)
        assert ecg.dtype==np.float32 and hrv.dtype==np.float32
        assert abs(float(ecg.mean()))<1e-4 and abs(float(ecg.std())-1)<1e-4
        missing+=np.sum(~np.isfinite(hrv),axis=0)
        per_patient.setdefault(row['patient'],Counter())[row['label']]+=1
    report=dict(windows=len(rows),patients=len(per_patient),labels=dict(Counter(r['label'] for r in rows)),
        per_patient={k:dict(v) for k,v in per_patient.items()},
        per_dataset={d:dict(Counter(r['label'] for r in rows if r['dataset']==d)) for d in ('afdb','ltafdb')},
        hrv_missing_fraction=dict(zip(HRV_ORDER,(missing/(20*len(rows))).tolist())),
        afdb_label_changes=dict(label_changes),exclusions=excluded,duplicate_source_recordings=0,
        label_policy='onset-v2',identity_limitation='Namespaced record grouping; cross-dataset person linkage unavailable',
        pilot_manifest_sha256=digest(pilot/'manifest.json'),preprocessing_sha256=digest(Path(__file__).with_name('preprocess.py')))
    (out/'cohort-audit.json').write_text(json.dumps(report,indent=2))
    # Publish manifest last, only after all source and tensor audits passed.
    (out/'manifest.json').write_text(json.dumps(rows,indent=2))
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--pilot-root',required=True)
    parser.add_argument('--ltaf-root',required=True)
    parser.add_argument('--out-root',required=True)
    args=parser.parse_args()
    build(Path(args.pilot_root),Path(args.ltaf_root),external_root(args.out_root))

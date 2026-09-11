"""Assemble complete AFDB outputs after an interrupted preprocessing run."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from afib.preprocess import external_root


def assemble(root):
    root = Path(root)
    checksums = {line.split()[1].lstrip('*'):line.split()[0]
                 for line in (root/'raw'/'SHA256SUMS').read_text().splitlines()}
    records = sorted(Path(name).stem for name in checksums if name.endswith('.dat') and '/' not in name)
    if not records:
        raise ValueError('No signal-bearing records in source checksum inventory')
    pipeline_hash = hashlib.sha256(Path(__file__).with_name('preprocess.py').read_bytes()).hexdigest()
    rows, keys = [], set()
    for record in records:
        directory = root/'processed'/record
        if not (directory/'report.json').exists() or not (directory/'manifest.json').exists():
            raise ValueError(f'Incomplete preprocessing: {record}; existing manifest retained')
        report = json.loads((directory/'report.json').read_text())
        part = json.loads((directory/'manifest.json').read_text())
        if report['preprocessing_sha256'] != pipeline_hash:
            raise ValueError(f'Preprocessing version mismatch: {record}')
        for extension in ('hea', 'dat', 'atr'):
            name = f'{record}.{extension}'
            digest = hashlib.sha256((root/'raw'/name).read_bytes()).hexdigest()
            if digest != checksums[name] or digest != report['source_sha256'][name]:
                raise ValueError(f'Source integrity mismatch: {name}')
        if len(part) != report['accepted']:
            raise ValueError(f'Record count mismatch: {record}')
        counts = {str(k):v for k,v in Counter(r['label'] for r in part).items()}
        if counts != report['labels']:
            raise ValueError(f'Label count mismatch: {record}')
        for row in part:
            key = (row['record'], row['start_sample'])
            if key in keys or row['record'] != record or row['patient'] != record:
                raise ValueError(f'Duplicate or inconsistent identity: {key}')
            path = Path(row['path']).resolve()
            if path.parent != directory.resolve() or not path.is_file():
                raise ValueError(f'Missing or misplaced artifact: {path}')
            keys.add(key)
        rows.extend(part)
    rows.sort(key=lambda row: (row['record'], row['start_sample']))
    temporary = root/'manifest.assembling.json'
    temporary.write_text(json.dumps(rows, indent=2))
    temporary.replace(root/'manifest.json')
    print(json.dumps(dict(records_completed=len(records), windows=len(rows),
                          labels=dict(Counter(r['label'] for r in rows)), manifest=str(root/'manifest.json'))))
    return rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    assemble(external_root(parser.parse_args().root))

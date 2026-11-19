"""Verify immutable release bytes and the published sealed-test confusion counts."""
from pathlib import Path
import csv
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]

def main():
    manifest = json.loads((ROOT/'artifacts/RELEASE_MANIFEST.json').read_text())
    for entry in manifest['files']:
        path = ROOT/entry['path']
        data = path.read_bytes()
        assert len(data) == entry['bytes'], f'Size mismatch: {path}'
        assert hashlib.sha256(data).hexdigest() == entry['sha256'], f'Hash mismatch: {path}'
    freeze = json.loads((ROOT/'artifacts/frozen_forecast_v1/FORECAST_FREEZE.json').read_text())
    for name in ('forecast_checkpoint.pt','forecast_summary.json','forecast_validation_predictions.csv'):
        data = (ROOT/'artifacts/frozen_forecast_v1'/name).read_bytes()
        assert hashlib.sha256(data).hexdigest() == freeze['files'][name]['sha256']
    summary = json.loads((ROOT/'results/frozen_forecast_sealed_test_v1/summary.json').read_text())
    metadata = json.loads((ROOT/'deployment/pi/model/deployment_metadata.json').read_text())
    assert summary['model_sha256'] == metadata['source_checkpoint_sha256']
    assert summary['overall']['threshold'] == metadata['validation_derived_cutoff'] == 0.3310546875
    rows = list(csv.DictReader((ROOT/'results/frozen_forecast_sealed_test_v1/test_predictions.csv').open()))
    counts = dict(TP=0,FN=0,FP=0,TN=0)
    for row in rows:
        positive = float(row['probability']) >= summary['overall']['threshold']
        actual = int(row['label']) == 1
        counts['TP' if actual and positive else 'FN' if actual else 'FP' if positive else 'TN'] += 1
    assert len(rows) == summary['overall']['windows']
    assert all(summary['overall'][k] == v for k,v in counts.items()), counts
    validation = list(csv.DictReader((ROOT/'artifacts/frozen_forecast_v1/forecast_validation_predictions.csv').open()))
    assert not ({r['subject_id'] for r in validation} & {r['subject_id'] for r in rows}), 'Validation/test patient overlap'
    print(f"PASS: {len(manifest['files'])} release files; sealed-test counts {counts}")

if __name__ == '__main__':
    main()

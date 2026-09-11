import hashlib
import json
from pathlib import Path

import pytest

from afib.manifest import assemble


def test_incomplete_assembly_preserves_existing_manifest(tmp_path):
    (tmp_path/'raw').mkdir()
    (tmp_path/'raw'/'SHA256SUMS').write_text('abc  00001.dat\n')
    target = tmp_path/'manifest.json'
    target.write_text('existing manifest')
    with pytest.raises(ValueError, match='Incomplete preprocessing'):
        assemble(tmp_path)
    assert target.read_text() == 'existing manifest'


def test_assembly_verifies_sources_and_accepts_zero_eligible_record(tmp_path):
    raw, processed = tmp_path/'raw', tmp_path/'processed'/'00001'
    raw.mkdir()
    processed.mkdir(parents=True)
    hashes = {}
    for extension in ('dat', 'hea', 'atr'):
        name = f'00001.{extension}'
        content = f'synthetic unit-test fixture {extension}'.encode()
        (raw/name).write_bytes(content)
        hashes[name] = hashlib.sha256(content).hexdigest()
    (raw/'SHA256SUMS').write_text(''.join(f'{digest}  {name}\n' for name,digest in hashes.items()))
    pipeline = Path(__file__).resolve().parents[1]/'afib'/'preprocess.py'
    report = dict(preprocessing_sha256=hashlib.sha256(pipeline.read_bytes()).hexdigest(),
                  source_sha256=hashes, accepted=0, labels={})
    (processed/'report.json').write_text(json.dumps(report))
    (processed/'manifest.json').write_text('[]')
    assert assemble(tmp_path) == []
    (raw/'00001.dat').write_bytes(b'corrupted')
    with pytest.raises(ValueError, match='Source integrity mismatch'):
        assemble(tmp_path)

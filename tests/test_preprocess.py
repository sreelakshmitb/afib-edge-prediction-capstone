from types import SimpleNamespace

import numpy as np
import pytest

from afib.preprocess import label_window, rhythm_intervals, hrv_features, preprocess_observation, external_root, download_mirror


def test_rhythm_carries_state_and_unknown_prefix():
    ann = SimpleNamespace(sample=[5, 100, 200], aux_note=['(N', '(AFIB', '(N'])
    assert rhythm_intervals(ann, 300) == [(0, 5, 'UNKNOWN'), (5, 100, '(N'), (100, 200, '(AFIB'), (200, 300, '(N')]


@pytest.mark.parametrize('onset, expected', [(599, None), (600, 1), (1800, 1), (1801, 0)])
def test_label_boundaries(onset, expected):
    intervals = [(0, onset, '(N'), (onset, 5000, '(AFIB')]
    assert label_window(0, 600, 1800, intervals, 5000) == expected


def test_incomplete_followup_and_competing_rhythm_excluded():
    assert label_window(0, 600, 1800, [(0, 1800, '(N')], 1800) is None
    assert label_window(0, 600, 1800, [(0, 700, '(N'), (700, 5000, '(AFL')], 5000) is None


def test_hrv_units_order_and_future_invariance():
    peaks = np.arange(0, 600 * 250, 250)
    hrv = hrv_features(peaks, 30)
    np.testing.assert_allclose(hrv[[0, 1, 3, 4]], [0, 0, 1000, 0])
    assert np.isnan(hrv[2]) and np.isnan(hrv[5])
    np.testing.assert_equal(hrv, hrv_features(peaks[peaks < 30*250], 30))


def test_bad_signal_rejected():
    with pytest.raises(ValueError, match='flat_signal'):
        preprocess_observation(np.zeros(150000), 250)
    with pytest.raises(ValueError, match='nonfinite'):
        preprocess_observation(np.full(150000, np.nan), 250)


def test_data_cannot_be_written_in_checkout():
    with pytest.raises(ValueError, match='outside'):
        external_root('.')


def test_resumable_download_validates_byte_range(monkeypatch, tmp_path):
    class Response:
        status_code = 206
        headers = {'Content-Length':'6', 'Content-Range':'bytes 3-5/6'}
        content = b'def'
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def raise_for_status(self): pass
    monkeypatch.setattr('afib.preprocess.requests.head', lambda *a, **k: Response())
    def get(*args, **kwargs):
        assert kwargs['headers']['Range'] == 'bytes=3-5'
        return Response()
    monkeypatch.setattr('afib.preprocess.requests.get', get)
    (tmp_path / 'record.partial').write_bytes(b'abc')
    download_mirror('record', tmp_path)
    assert (tmp_path / 'record').read_bytes() == b'abcdef'

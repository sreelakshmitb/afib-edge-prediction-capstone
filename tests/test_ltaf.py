from types import SimpleNamespace

import numpy as np

from afib.ltaf import merged_intervals, onset_label
from afib.preprocess import label_window, preprocess_observation


def test_repeated_rhythm_markers_are_not_new_onsets():
    ann = SimpleNamespace(sample=[0, 100, 200, 300, 400],
                          aux_note=['(N', '(AFIB', '(AFIB', 'MISSB', '(N'])
    assert merged_intervals(ann, 500) == [(0, 100, '(N'), (100, 400, '(AFIB'), (400, 500, '(N')]


def test_native_128hz_onset_boundary():
    end, horizon = 600*128, 1800*128
    assert label_window(0, end, horizon, [(0,end,'(N'), (end,300000,'(AFIB')],300000)==1
    assert label_window(0, end, horizon, [(0,end-1,'(N'), (end-1,300000,'(AFIB')],300000) is None


def test_other_known_future_rhythm_does_not_erase_af_outcome():
    intervals = [(0,700,'(N'), (700,800,'(VT'), (800,2000,'(AFIB')]
    assert onset_label(0,600,1800,intervals,2000)==1
    assert label_window(0,600,1800,intervals,2000) is None
    assert onset_label(0,600,1800,[(0,700,'(N'),(700,2000,'UNKNOWN')],2000) is None
    assert onset_label(0,600,1800,[(0,700,'(N'),(800,2000,'(N')],2000) is None
    assert onset_label(0,600,1800,[(0,1800,'(N'),(1800,2000,'(AFIB')],2000)==1


def test_128hz_resampling_and_hrv_contract():
    # Deterministic synthetic QRS train checks rate/shape, not clinical R-peak accuracy.
    t = np.arange(128*600)/128
    signal = .01*np.sin(2*np.pi*3*t)
    phase = (t-.4+.5)%1-.5
    signal += np.exp(-.5*(phase/.015)**2)
    ecg, hrv, peaks = preprocess_observation(signal, 128)
    assert ecg.shape == (20,1,7500) and hrv.shape == (20,6)
    assert ecg.dtype == np.float32 and np.isfinite(ecg).all()
    assert 590 <= peaks <= 610
    assert np.nanmedian(hrv[:,3]) > 990
    assert np.isnan(hrv[:9,2]).all()

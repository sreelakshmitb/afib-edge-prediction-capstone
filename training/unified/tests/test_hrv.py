import numpy as np

from afib_unified.hrv_preprocess import HRVFoldPreprocessor


def test_hrv_fit_imputes_and_scales_from_training_values():
    x = np.arange(240, dtype=np.float32).reshape(2, 20, 6)
    x[0, 0, 2] = np.nan
    processor = HRVFoldPreprocessor().fit(x)
    transformed = processor.transform(x)
    assert transformed.shape == x.shape
    assert np.isfinite(transformed).all()
    assert processor.median_.shape == (6,)

import numpy as np
import torch

from afib.data import HRVTransform
from afib.export import InferenceModel
from afib.models import AFibModel


def test_export_embeds_identical_hrv_transform():
    torch.set_num_threads(2)
    values = np.arange(120, dtype=np.float32).reshape(1,20,6)
    values[0, :10, 2] = np.nan
    values[0, 5, 5] = np.inf
    transform = HRVTransform().fit(values)
    model = AFibModel().eval()
    wrapper = InferenceModel(model, transform.state()).eval()
    ecg = torch.randn(1,20,1,7500)
    with torch.no_grad():
        expected = torch.sigmoid(model(ecg, torch.from_numpy(transform(values))))
        actual = wrapper(ecg, torch.from_numpy(values))
    torch.testing.assert_close(actual, expected)

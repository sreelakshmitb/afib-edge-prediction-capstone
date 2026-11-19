import torch

from afib_finalization.model import CurrentAFDetector


def test_detector_contract_and_budget():
    model = CurrentAFDetector()
    y = model(torch.zeros(2, 1, 7500))
    assert y.shape == (2,)
    assert model.parameter_count < 50_000
    assert model.parameter_count + 159_503 < 250_000

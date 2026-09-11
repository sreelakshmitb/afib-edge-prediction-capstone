import numpy as np
import pytest
import torch

from afib.data import HRVTransform, patient_splits
from afib.models import AFibModel


def test_patient_separation_and_outer_coverage():
    rows = [dict(patient=str(p), label=i%2) for p in range(10) for i in range(4)]
    all_test = []
    for _, train, val, test in patient_splits(rows):
        sets = [{rows[i]['patient'] for i in indices} for indices in (train, val, test)]
        assert not sets[0] & sets[1] and not sets[0] & sets[2] and not sets[1] & sets[2]
        all_test.extend(test)
    assert sorted(all_test) == list(range(len(rows)))


def test_transform_does_not_fit_on_validation_and_preserves_missing_column():
    train = np.arange(120, dtype=float).reshape(20, 6)
    train[:, 2] = np.nan
    transform = HRVTransform().fit(train)
    before = transform.state()
    assert np.isfinite(transform(np.full((20, 6), np.nan))).all()
    transform(np.full((20, 6), 1e12))
    assert before == transform.state()
    np.testing.assert_equal(transform(train), HRVTransform.from_state(before)(train))


@pytest.mark.parametrize('se', [False, True])
def test_model_full_contract_backward_and_budget(se):
    torch.set_num_threads(2)
    model = AFibModel(se=se)
    assert sum(p.numel() for p in model.parameters()) < 250000
    output = model(torch.randn(2, 20, 1, 7500), torch.randn(2, 20, 6))
    assert output.shape == (2,)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(output, torch.tensor([0., 1.]))
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    with pytest.raises(ValueError, match='ECG contract'):
        model(torch.randn(2, 1, 7500), torch.randn(2, 20, 6))

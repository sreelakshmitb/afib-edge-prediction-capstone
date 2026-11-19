import torch
from torch import nn

from afib_unified.models import build_model


def test_proposed_model_contract_and_budget():
    model = build_model("proposed")
    assert model.parameter_count == 159503
    with torch.no_grad():
        logits = model(torch.zeros(2, 20, 1, 7500), torch.zeros(2, 20, 6))
    assert logits.shape == (2,)
    assert torch.isfinite(logits).all()


def test_groupnorm_repair_preserves_contract_and_budget():
    model = build_model("proposed-groupnorm")
    assert model.parameter_count == 159503
    assert any(isinstance(module, nn.GroupNorm) for module in model.modules())
    assert not any(isinstance(module, nn.BatchNorm1d) for module in model.modules())
    with torch.no_grad():
        logits = model(torch.zeros(2, 20, 1, 7500), torch.zeros(2, 20, 6))
    assert logits.shape == (2,)
    assert torch.isfinite(logits).all()
    with torch.no_grad():
        logits, representation = model.forward_with_representation(
            torch.zeros(2, 20, 1, 7500), torch.zeros(2, 20, 6)
        )
    assert logits.shape == (2,)
    assert representation.shape == (2, 128)


def test_multihorizon_model_is_monotonic_and_within_budget():
    model = build_model("proposed-groupnorm-multihorizon")
    assert model.parameter_count == 160019
    with torch.no_grad():
        logits = model(torch.zeros(2, 20, 1, 7500), torch.zeros(2, 20, 6))
        risks = model.cumulative_risk_from_logits(logits)
    assert logits.shape == (2, 5)
    assert risks.shape == (2, 5)
    assert torch.isfinite(risks).all()
    assert torch.all(risks[:, 1:] >= risks[:, :-1])

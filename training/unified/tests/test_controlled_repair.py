import torch

from controlled_repair import (
    apply_input_mode,
    conditional_domain_mean_loss,
    input_mode_for_recipe,
    lead_bins_for_epoch,
)


def test_branch_recipe_modes_and_curriculum():
    assert input_mode_for_recipe("groupnorm-curriculum-ecg-only") == "ecg-only"
    assert input_mode_for_recipe("groupnorm-curriculum-hrv-only") == "hrv-only"
    assert lead_bins_for_epoch("groupnorm-curriculum-ecg-only", 0) == ("00_04", "04_08")
    assert len(lead_bins_for_epoch("groupnorm-curriculum-hrv-only", 6)) == 5


def test_input_masking_is_exact():
    ecg = torch.ones(2, 20, 1, 8)
    hrv = torch.ones(2, 20, 6)
    ecg_only = apply_input_mode(ecg, hrv, "ecg-only")
    hrv_only = apply_input_mode(ecg, hrv, "hrv-only")
    assert torch.equal(ecg_only[0], ecg)
    assert torch.count_nonzero(ecg_only[1]) == 0
    assert torch.count_nonzero(hrv_only[0]) == 0
    assert torch.equal(hrv_only[1], hrv)


def test_conditional_domain_alignment_detects_source_shift_and_has_gradient():
    labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    sources = torch.tensor([False, False, True, True] * 2)
    representation = torch.zeros(8, 4, requires_grad=True)
    with torch.no_grad():
        representation[sources] += 1.0
    loss = conditional_domain_mean_loss(representation, labels, sources)
    assert loss > 0
    loss.backward()
    assert representation.grad is not None
    assert torch.isfinite(representation.grad).all()

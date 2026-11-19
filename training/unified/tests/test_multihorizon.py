import numpy as np
import pandas as pd
import torch

from afib_unified.multihorizon import (
    balanced_discrete_time_nll,
    build_multihorizon_manifest,
    cumulative_risk_from_logits,
    cumulative_targets,
    lead_bin_index,
)


def test_exact_lead_boundaries_and_cumulative_targets():
    leads = np.asarray([0.001, 4.0, 4.001, 8.0, 8.001, 12.0, 12.001, 16.0, 16.001, 20.0])
    assert lead_bin_index(leads).tolist() == [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]
    labels = np.asarray([1, 1, 1, 1, 1, 0])
    bins = np.asarray([0, 1, 2, 3, 4, -1])
    targets = cumulative_targets(labels, bins)
    assert targets.tolist() == [
        [1, 1, 1, 1, 1],
        [0, 1, 1, 1, 1],
        [0, 0, 1, 1, 1],
        [0, 0, 0, 1, 1],
        [0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0],
    ]


def test_balanced_survival_loss_rewards_correct_interval_and_has_gradient():
    bins = torch.tensor([0, 4, -1])
    correct = torch.tensor(
        [[6.0, -6.0, -6.0, -6.0, -6.0],
         [-6.0, -6.0, -6.0, -6.0, 6.0],
         [-6.0, -6.0, -6.0, -6.0, -6.0]],
        requires_grad=True,
    )
    wrong = -correct.detach()
    good_loss = balanced_discrete_time_nll(correct, bins)
    bad_loss = balanced_discrete_time_nll(wrong, bins)
    assert good_loss < bad_loss
    good_loss.backward()
    assert correct.grad is not None
    assert torch.isfinite(correct.grad).all()
    risks = cumulative_risk_from_logits(correct.detach())
    assert torch.all(risks[:, 1:] >= risks[:, :-1])


def test_manifest_reconstruction_preserves_cache_addressing(tmp_path):
    cached = pd.DataFrame(
        {
            "manifest_index": [0, 1],
            "cache_family": ["legacy", "iridia"],
            "cache_local_index": [0, 0],
            "cache_record_id": ["legacy:p0", "record_0"],
            "dataset": ["afdb", "iridia"],
            "event_id": ["legacy:e0", np.nan],
            "label": [1, 0],
            "lead_bin": ["04_08", np.nan],
            "negative_stratum": ["legacy_unstratified", "far_360_plus_or_none"],
            "record_id": ["r0", "record_0"],
            "source_family": ["legacy", "iridia"],
            "start_sample_250hz": [100, 200],
            "subject_id": ["legacy:p0", "iridia:p0"],
            "training_eligible": [True, True],
            "window_id": ["legacy:p0:100", "iridia:record_0:200"],
            "outer_test_fold": [0, 1],
        }
    )
    combined = pd.DataFrame(
        {
            "source_family": ["legacy", "iridia"],
            "record_id": ["r0", "record_0"],
            "subject_id": ["legacy:p0", "iridia:p0"],
            "start_sample_250hz": [100, 200],
            "label": [1, 0],
            "lead_bin": ["04_08", np.nan],
            "event_id": ["legacy:e0", np.nan],
            "observation_start_seconds": [0.4, 0.8],
            "forecast_start_seconds": [600.4, 600.8],
            "forecast_end_seconds": [1800.4, 1800.8],
            "end_sample_250hz": [150100, 150200],
            "horizon_end_sample_250hz": [450100, 450200],
            "af_onset_seconds": [900.4, np.nan],
            "lead_time_minutes": [5.0, np.nan],
            "next_af_onset_minutes": [5.0, 120.0],
            "minutes_since_af_end": [np.nan, 80.0],
        }
    )
    cached_path = tmp_path / "cached.csv"
    combined_path = tmp_path / "combined.csv"
    cached.to_csv(cached_path, index=False)
    combined.to_csv(combined_path, index=False)
    result = build_multihorizon_manifest(cached_path, combined_path)
    assert result.manifest_index.tolist() == [0, 1]
    assert result.event_bin.tolist() == [1, -1]
    assert result[["target_04", "target_08", "target_12", "target_16", "target_20"]].values.tolist() == [
        [0, 1, 1, 1, 1], [0, 0, 0, 0, 0]
    ]

import pandas as pd

from afib_unified.contracts import LEAD_BINS
from afib_unified.sampling import HierarchicalSamplerConfig, build_epoch_schedule


def test_exact_balanced_batch():
    rows = []
    index = 0
    for family in ("legacy", "iridia"):
        for label in (0, 1):
            for patient in range(30):
                rows.append({
                    "manifest_index": index,
                    "dataset": family,
                    "source_family": family,
                    "subject_id": f"{family}:p{patient}",
                    "label": label,
                    "event_id": f"{family}:e{patient}" if label else "",
                    "lead_bin": LEAD_BINS[patient % 5] if label else "",
                    "negative_stratum": (
                        ["near_20_60", "medium_60_360", "far_360_plus_or_none"][patient % 3]
                        if family == "iridia" and not label else "legacy_unstratified"
                    ),
                    "training_eligible": True,
                })
                index += 1
    schedule = build_epoch_schedule(
        pd.DataFrame(rows),
        HierarchicalSamplerConfig(batches_per_epoch=1, seed=7),
    )
    assert len(schedule) == 32
    assert schedule.label.sum() == 8
    counts = schedule.groupby(["label", "source_family"]).size().to_dict()
    assert counts == {(0, "iridia"): 12, (0, "legacy"): 12, (1, "iridia"): 4, (1, "legacy"): 4}
    assert not schedule.duplicated(["batch_id", "manifest_index"]).any()


def test_curriculum_restricts_positive_bins_without_changing_batch_ratio():
    rows = []
    index = 0
    for family in ("legacy", "iridia"):
        for label in (0, 1):
            for patient in range(30):
                rows.append({
                    "manifest_index": index,
                    "dataset": family,
                    "source_family": family,
                    "subject_id": f"{family}:p{patient}",
                    "label": label,
                    "event_id": f"{family}:e{patient}" if label else "",
                    "lead_bin": LEAD_BINS[patient % 5] if label else "",
                    "negative_stratum": (
                        ["near_20_60", "medium_60_360", "far_360_plus_or_none"][patient % 3]
                        if family == "iridia" and not label else "legacy_unstratified"
                    ),
                    "training_eligible": True,
                })
                index += 1
    allowed = ("00_04", "04_08")
    schedule = build_epoch_schedule(
        pd.DataFrame(rows),
        HierarchicalSamplerConfig(
            batches_per_epoch=3, positive_lead_bins=allowed, seed=7
        ),
    )
    positives = schedule[schedule.label.eq(1)]
    assert set(positives.lead_bin).issubset(set(allowed))
    assert schedule.groupby("batch_id").label.sum().eq(8).all()
    assert not schedule.duplicated(["batch_id", "manifest_index"]).any()


def test_harmonized_sampler_stratifies_both_negative_sources():
    rows = []
    index = 0
    strata = ("near_20_60", "medium_60_360", "far_360_plus_or_none")
    for family in ("legacy", "iridia"):
        for patient in range(40):
            rows.append({
                "manifest_index": index,
                "dataset": family,
                "source_family": family,
                "subject_id": f"{family}:positive:{patient}",
                "label": 1,
                "event_id": f"{family}:event:{patient}",
                "lead_bin": LEAD_BINS[patient % 5],
                "negative_stratum": "",
                "training_eligible": True,
            })
            index += 1
        for patient in range(60):
            rows.append({
                "manifest_index": index,
                "dataset": family,
                "source_family": family,
                "subject_id": f"{family}:negative:{patient}",
                "label": 0,
                "event_id": "",
                "lead_bin": "",
                "negative_stratum": strata[patient % 3],
                "training_eligible": True,
            })
            index += 1
    schedule = build_epoch_schedule(
        pd.DataFrame(rows),
        HierarchicalSamplerConfig(
            batches_per_epoch=5,
            stratified_negative_sources=("iridia", "legacy"),
            seed=7,
        ),
    )
    negatives = schedule[schedule.label.eq(0)]
    counts = negatives.groupby(["source_family", "negative_stratum"]).size().to_dict()
    assert counts == {
        ("iridia", "far_360_plus_or_none"): 30,
        ("iridia", "medium_60_360"): 18,
        ("iridia", "near_20_60"): 12,
        ("legacy", "far_360_plus_or_none"): 30,
        ("legacy", "medium_60_360"): 18,
        ("legacy", "near_20_60"): 12,
    }
    assert not schedule.duplicated(["batch_id", "manifest_index"]).any()

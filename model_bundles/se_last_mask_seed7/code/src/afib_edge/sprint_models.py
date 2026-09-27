"""Models and training switches for prediction quality sprints.

Every SE candidate uses the original last-state SE-CNN-HRV-UniLSTM. Training
recipes may change, but deployment parameter counts and input contracts remain
unchanged.
"""

from __future__ import annotations

from torch import nn

from .models import ModelConfig, build_model


SE_CANDIDATES = frozenset(
    {
        "se_last_natural",
        "se_last_mask",
        "se_last_mask_ap",
        "se_last_mask_curriculum",
        "se_last_mask_patient_equal",
        "se_last_mask_patient_equal_ap",
        "se_last_mask_afpdb",
        "se_last_mask_icentia",
        "se_last_focal",
        "se_last_focal_mask",
    }
)
MASKED_CANDIDATES = frozenset(
    {
        "se_last_mask",
        "se_last_mask_ap",
        "se_last_mask_curriculum",
        "se_last_mask_patient_equal",
        "se_last_mask_patient_equal_ap",
        "se_last_mask_afpdb",
        "se_last_mask_icentia",
        "se_last_focal_mask",
    }
)
FOCAL_CANDIDATES = frozenset({"se_last_focal", "se_last_focal_mask"})
ICENTIA_PRETRAINED_CANDIDATES = frozenset({"se_last_mask_icentia"})
AFPDB_PRETRAINED_CANDIDATES = frozenset({"se_last_mask_afpdb"})
HORIZON_CURRICULUM_CANDIDATES = frozenset({"se_last_mask_curriculum"})
PATIENT_EQUAL_CANDIDATES = frozenset(
    {"se_last_mask_patient_equal", "se_last_mask_patient_equal_ap"}
)
AP_ALIGNED_CANDIDATES = frozenset(
    {"se_last_mask_ap", "se_last_mask_patient_equal_ap"}
)


def uses_ecg_masking(candidate: str) -> bool:
    return candidate in MASKED_CANDIDATES


def uses_focal_loss(candidate: str) -> bool:
    return candidate in FOCAL_CANDIDATES


def uses_icentia_pretraining(candidate: str) -> bool:
    return candidate in ICENTIA_PRETRAINED_CANDIDATES


def uses_afpdb_pretraining(candidate: str) -> bool:
    return candidate in AFPDB_PRETRAINED_CANDIDATES


def uses_horizon_curriculum(candidate: str) -> bool:
    return candidate in HORIZON_CURRICULUM_CANDIDATES


def uses_patient_equal_loss(candidate: str) -> bool:
    return candidate in PATIENT_EQUAL_CANDIDATES


def uses_ap_aligned_loss(candidate: str) -> bool:
    return candidate in AP_ALIGNED_CANDIDATES


def build_sprint_model(candidate: str, dropout: float = 0.3) -> nn.Module:
    if candidate == "paper_control":
        model = build_model("paper", ModelConfig(dropout=dropout))
    elif candidate in SE_CANDIDATES:
        model = build_model("proposed", ModelConfig(dropout=dropout))
    else:
        raise ValueError(f"unknown neural candidate: {candidate}")
    if model.parameter_count >= 250_000:
        raise ValueError("candidate exceeds the project parameter budget")
    return model


__all__ = [
    "AFPDB_PRETRAINED_CANDIDATES",
    "AP_ALIGNED_CANDIDATES",
    "FOCAL_CANDIDATES",
    "HORIZON_CURRICULUM_CANDIDATES",
    "ICENTIA_PRETRAINED_CANDIDATES",
    "MASKED_CANDIDATES",
    "PATIENT_EQUAL_CANDIDATES",
    "SE_CANDIDATES",
    "build_sprint_model",
    "uses_ecg_masking",
    "uses_focal_loss",
    "uses_horizon_curriculum",
    "uses_icentia_pretraining",
    "uses_afpdb_pretraining",
    "uses_ap_aligned_loss",
    "uses_patient_equal_loss",
]

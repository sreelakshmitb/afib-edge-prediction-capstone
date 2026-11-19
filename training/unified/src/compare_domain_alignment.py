"""Compare the full curriculum model with label-conditional domain alignment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score

from afib_unified.manifest import sha256_file
from afib_unified.validation_metrics import attach_manifest_metadata, summarize_predictions


RUNS = {
    "full_fusion": "groupnorm_curriculum",
    "conditional_domain_alignment": "groupnorm_curriculum_domain_align",
}


def zero_recall_fraction(diagnostics: dict, source: str) -> float:
    values = diagnostics["by_source"][source]["patient_macro_at_source_threshold"]
    return values["zero_recall_patients"] / values["positive_patients"]


def compact(diagnostics: dict, predictions: pd.DataFrame) -> dict:
    negative = predictions.label.eq(0)
    source_labels = predictions.loc[negative, "source_family"].eq("iridia").astype(int)
    source_auc = roc_auc_score(source_labels, predictions.loc[negative, "probability"])
    return {
        "pr_auc": diagnostics["ranking"]["pr_auc"],
        "auroc": diagnostics["ranking"]["auroc"],
        "recall_at_90_spec": diagnostics["global_operating_point"]["recall"],
        "specificity": diagnostics["global_operating_point"]["specificity"],
        "iridia_pr_auc": diagnostics["by_source"]["iridia"]["ranking"]["pr_auc"],
        "iridia_auroc": diagnostics["by_source"]["iridia"]["ranking"]["auroc"],
        "legacy_pr_auc": diagnostics["by_source"]["legacy"]["ranking"]["pr_auc"],
        "legacy_auroc": diagnostics["by_source"]["legacy"]["ranking"]["auroc"],
        "iridia_event_sensitivity": diagnostics["by_source"]["iridia"]
        ["event_sensitivity_at_source_threshold"]["sensitivity"],
        "legacy_event_sensitivity": diagnostics["by_source"]["legacy"]
        ["event_sensitivity_at_source_threshold"]["sensitivity"],
        "iridia_zero_recall_patient_fraction": zero_recall_fraction(diagnostics, "iridia"),
        "legacy_zero_recall_patient_fraction": zero_recall_fraction(diagnostics, "legacy"),
        "negative_source_score_separability_auc": max(source_auc, 1.0 - source_auc),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--protocol-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest_path = args.protocol_dir / "unified_cached_manifest.csv"
    protocol = json.loads((args.protocol_dir / "protocol_summary.json").read_text())
    if sha256_file(manifest_path) != protocol["manifest_sha256"]:
        raise ValueError("manifest hash mismatch")
    manifest = pd.read_csv(manifest_path)
    results = {}
    for name, folder in RUNS.items():
        directory = args.runs_root / folder
        summary = json.loads((directory / "controlled_repair_summary.json").read_text())
        if not summary.get("all_gates_pass"):
            raise RuntimeError(f"integrity gates failed: {name}")
        predictions = attach_manifest_metadata(
            pd.read_csv(directory / "best_validation_predictions.csv"), manifest
        )
        diagnostics = summarize_predictions(predictions)
        results[name] = {
            "best_epoch": summary["best_epoch"],
            "metrics": compact(diagnostics, predictions),
        }
    reference = results["full_fusion"]["metrics"]
    candidate = results["conditional_domain_alignment"]["metrics"]
    mechanism_gates = {
        "negative_source_separability_reduced_by_0_03": (
            candidate["negative_source_score_separability_auc"]
            <= reference["negative_source_score_separability_auc"] - 0.03
        ),
        "overall_pr_auc_not_reduced": candidate["pr_auc"] >= reference["pr_auc"],
        "legacy_auroc_at_least_0_60": candidate["legacy_auroc"] >= 0.60,
        "iridia_auroc_at_least_0_64": candidate["iridia_auroc"] >= 0.64,
        "recall_at_90_spec_at_least_0_35": candidate["recall_at_90_spec"] >= 0.35,
    }
    mechanism_gates = {
        key: bool(value) for key, value in mechanism_gates.items()
    }
    clinical_progression_gates = {
        "overall_pr_auc_at_least_0_267468": candidate["pr_auc"] >= 0.2674680603438324,
        "recall_at_90_spec_at_least_0_40": candidate["recall_at_90_spec"] >= 0.40,
        "legacy_auroc_at_least_0_60": candidate["legacy_auroc"] >= 0.60,
        "iridia_auroc_at_least_0_64": candidate["iridia_auroc"] >= 0.64,
    }
    clinical_progression_gates = {
        key: bool(value) for key, value in clinical_progression_gates.items()
    }
    output = {
        "format": "afib_conditional_domain_alignment_comparison_v1",
        "results": results,
        "mechanism_gates": mechanism_gates,
        "passes_mechanism_gate": all(mechanism_gates.values()),
        "clinical_progression_gates": clinical_progression_gates,
        "passes_clinical_progression_gate": all(clinical_progression_gates.values()),
        "outer_test_unlock": False,
        "outer_test_evaluated": False,
        "manifest_sha256": protocol["manifest_sha256"],
        "splits_sha256": protocol["splits_sha256"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True))
    print("CONDITIONAL DOMAIN-ALIGNMENT COMPARISON")
    print("=" * 88)
    for name, result in results.items():
        metrics = result["metrics"]
        print(
            f"{name}: best_epoch={result['best_epoch']}; PR-AUC={metrics['pr_auc']:.4f}; "
            f"AUROC={metrics['auroc']:.4f}; recall@90spec={metrics['recall_at_90_spec']:.3f}; "
            f"IRIDIA AUROC={metrics['iridia_auroc']:.4f}; "
            f"legacy AUROC={metrics['legacy_auroc']:.4f}; "
            f"negative-source-separability="
            f"{metrics['negative_source_score_separability_auc']:.4f}"
        )
    print("\nMECHANISM GATES")
    for key, value in mechanism_gates.items():
        print(f"{key}: {value}")
    print("passes_mechanism_gate:", output["passes_mechanism_gate"])
    print("\nCLINICAL PROGRESSION GATES")
    for key, value in clinical_progression_gates.items():
        print(f"{key}: {value}")
    print("passes_clinical_progression_gate:", output["passes_clinical_progression_gate"])
    print("outer_test_unlock: False")
    print("No outer-test patients were evaluated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

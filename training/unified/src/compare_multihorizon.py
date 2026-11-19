"""Compare the structural multi-horizon model with the frozen single-head baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from afib_unified.manifest import sha256_file
from afib_unified.validation_metrics import summarize_predictions


REQUIRED_PR_AUC = 0.2674680603438324


def compact(diagnostics: dict) -> dict:
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
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-run", type=Path, required=True)
    parser.add_argument("--multihorizon-run", type=Path, required=True)
    parser.add_argument("--target-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    baseline_summary = json.loads(
        (args.baseline_run / "controlled_repair_summary.json").read_text()
    )
    candidate_summary = json.loads(
        (args.multihorizon_run / "multihorizon_summary.json").read_text()
    )
    target_summary = json.loads((args.target_dir / "target_summary.json").read_text())
    target_path = args.target_dir / "multihorizon_cached_manifest.csv"
    if not baseline_summary.get("all_gates_pass"):
        raise RuntimeError("single-head baseline integrity gates failed")
    if not candidate_summary.get("all_gates_pass"):
        raise RuntimeError("multi-horizon candidate integrity gates failed")
    if sha256_file(target_path) != target_summary["multihorizon_manifest_sha256"]:
        raise ValueError("target manifest hash mismatch")
    if candidate_summary["target_manifest_sha256"] != target_summary["multihorizon_manifest_sha256"]:
        raise ValueError("candidate used a different target manifest")

    baseline = pd.read_csv(args.baseline_run / "best_validation_predictions.csv")
    candidate = pd.read_csv(args.multihorizon_run / "best_validation_predictions.csv")
    if set(baseline.manifest_index.astype(int)) != set(candidate.manifest_index.astype(int)):
        raise ValueError("baseline and candidate validation windows differ")
    baseline_metrics = compact(summarize_predictions(baseline))
    candidate_20 = candidate[
        [
            "window_id", "subject_id", "source_family", "label", "event_id",
            "lead_bin", "negative_stratum", "risk_20",
        ]
    ].rename(columns={"risk_20": "probability"})
    candidate_metrics = compact(summarize_predictions(candidate_20))
    progression_gates = {
        "overall_pr_auc_at_least_0_267468": candidate_metrics["pr_auc"] >= REQUIRED_PR_AUC,
        "overall_pr_auc_exceeds_single_head": (
            candidate_metrics["pr_auc"] > baseline_metrics["pr_auc"]
        ),
        "recall_at_90_spec_at_least_0_40": candidate_metrics["recall_at_90_spec"] >= 0.40,
        "legacy_auroc_at_least_0_60": candidate_metrics["legacy_auroc"] >= 0.60,
        "iridia_auroc_at_least_0_64": candidate_metrics["iridia_auroc"] >= 0.64,
        "monotonic_risk_gate": candidate_summary["gates"]["cumulative_risk_monotonic"],
    }
    progression_gates = {key: bool(value) for key, value in progression_gates.items()}
    output = {
        "format": "afib_multihorizon_comparison_v1",
        "required_pr_auc": REQUIRED_PR_AUC,
        "single_head_baseline": baseline_metrics,
        "multihorizon_20min": candidate_metrics,
        "multihorizon_per_horizon": candidate_summary["per_horizon"],
        "progression_gates": progression_gates,
        "passes_progression_gate": all(progression_gates.values()),
        "outer_test_unlock": False,
        "outer_test_evaluated": False,
        "decision": (
            "candidate passed validation progression gates; preserve outer lock pending explicit review"
            if all(progression_gates.values())
            else "candidate did not pass every validation progression gate; outer evaluation remains locked"
        ),
        "target_manifest_sha256": target_summary["multihorizon_manifest_sha256"],
        "splits_sha256": candidate_summary["splits_sha256"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True))
    print("MULTI-HORIZON STRUCTURAL COMPARISON")
    print("=" * 88)
    for name, metrics in (
        ("single_head_baseline", baseline_metrics),
        ("multihorizon_20min", candidate_metrics),
    ):
        print(
            f"{name}: PR-AUC={metrics['pr_auc']:.4f}; AUROC={metrics['auroc']:.4f}; "
            f"recall@90spec={metrics['recall_at_90_spec']:.3f}; "
            f"IRIDIA AUROC={metrics['iridia_auroc']:.4f}; "
            f"legacy AUROC={metrics['legacy_auroc']:.4f}"
        )
    print("\nMULTI-HORIZON RANKING")
    for horizon, values in candidate_summary["per_horizon"].items():
        rank = values["ranking"]
        operating = values["operating_point"]
        print(
            f"{horizon} min: positives={rank['positives']}; PR-AUC={rank['pr_auc']:.4f}; "
            f"AUROC={rank['auroc']:.4f}; recall@90spec={operating['recall']:.3f}"
        )
    print("\nPROGRESSION GATES")
    for key, value in progression_gates.items():
        print(f"{key}: {value}")
    print("passes_progression_gate:", output["passes_progression_gate"])
    print("outer_test_unlock: False")
    print("decision:", output["decision"])
    print("No outer-test patients were evaluated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

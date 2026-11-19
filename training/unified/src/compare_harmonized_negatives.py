"""Compare timing-harmonized training with the same baseline on identical rows."""

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


def _harmonize_predictions(predictions: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    if "manifest_index" not in predictions.columns:
        raise ValueError("predictions lack manifest_index")
    metadata = manifest.set_index("manifest_index")
    indices = predictions.manifest_index.astype(int)
    if not set(indices).issubset(set(metadata.index)):
        raise ValueError("predictions contain unknown manifest indices")
    output = predictions.copy()
    output["evaluation_eligible"] = metadata.loc[
        indices, "evaluation_eligible"
    ].to_numpy()
    output["negative_stratum"] = metadata.loc[
        indices, "negative_stratum"
    ].fillna("").astype(str).to_numpy()
    eligible = output.evaluation_eligible.astype(str).str.lower().isin(["true", "1"])
    return output.loc[eligible].reset_index(drop=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-run", type=Path, required=True)
    parser.add_argument("--candidate-run", type=Path, required=True)
    parser.add_argument("--harmonized-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest_path = args.harmonized_dir / "harmonized_cached_manifest.csv"
    harmonized_summary = json.loads(
        (args.harmonized_dir / "harmonized_summary.json").read_text()
    )
    candidate_summary = json.loads(
        (args.candidate_run / "controlled_repair_summary.json").read_text()
    )
    if not harmonized_summary.get("all_gates_pass"):
        raise RuntimeError("harmonized manifest gates failed")
    if sha256_file(manifest_path) != harmonized_summary["harmonized_manifest_sha256"]:
        raise ValueError("harmonized manifest hash mismatch")
    if not candidate_summary.get("all_gates_pass"):
        raise RuntimeError("candidate integrity gates failed")
    if candidate_summary["manifest_sha256"] != harmonized_summary["harmonized_manifest_sha256"]:
        raise ValueError("candidate used a different harmonized manifest")
    manifest = pd.read_csv(manifest_path)
    raw_baseline = pd.read_csv(args.baseline_run / "best_validation_predictions.csv")
    baseline_all = summarize_predictions(raw_baseline)
    baseline = _harmonize_predictions(raw_baseline, manifest)
    candidate = _harmonize_predictions(
        pd.read_csv(args.candidate_run / "best_validation_predictions.csv"), manifest
    )
    if set(baseline.manifest_index.astype(int)) != set(candidate.manifest_index.astype(int)):
        raise ValueError("baseline and candidate are not evaluated on identical corrected rows")
    baseline_diagnostics = summarize_predictions(baseline)
    candidate_diagnostics = summarize_predictions(candidate)
    baseline_metrics = compact(baseline_diagnostics)
    candidate_metrics = compact(candidate_diagnostics)
    gates = {
        "overall_pr_auc_at_least_0_267468": candidate_metrics["pr_auc"] >= REQUIRED_PR_AUC,
        "overall_pr_auc_exceeds_corrected_baseline": (
            candidate_metrics["pr_auc"] > baseline_metrics["pr_auc"]
        ),
        "recall_at_90_spec_at_least_0_40": candidate_metrics["recall_at_90_spec"] >= 0.40,
        "legacy_auroc_at_least_0_60": candidate_metrics["legacy_auroc"] >= 0.60,
        "iridia_auroc_at_least_0_64": candidate_metrics["iridia_auroc"] >= 0.64,
        "same_corrected_validation_rows": set(baseline.manifest_index.astype(int))
        == set(candidate.manifest_index.astype(int)),
    }
    gates = {key: bool(value) for key, value in gates.items()}
    validation_subjects = set(raw_baseline.subject_id.astype(str))
    excluded_validation = manifest[
        manifest.subject_id.astype(str).isin(validation_subjects)
        & ~manifest.evaluation_eligible.astype(str).str.lower().isin(["true", "1"])
    ]
    output = {
        "format": "afib_harmonized_negative_comparison_v1",
        "required_pr_auc": REQUIRED_PR_AUC,
        "original_baseline_all_rows": compact(baseline_all),
        "corrected_baseline_same_rows": baseline_metrics,
        "harmonized_candidate": candidate_metrics,
        "corrected_validation_windows": int(len(candidate)),
        "excluded_validation_post_af_windows": int(len(excluded_validation)),
        "progression_gates": gates,
        "passes_progression_gate": all(gates.values()),
        "outer_test_unlock": False,
        "outer_test_evaluated": False,
        "decision": (
            "candidate passed validation gates; preserve outer lock pending explicit review"
            if all(gates.values())
            else "candidate failed at least one validation gate; outer evaluation remains locked"
        ),
        "harmonized_manifest_sha256": harmonized_summary["harmonized_manifest_sha256"],
        "splits_sha256": candidate_summary["splits_sha256"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True))
    print("HARMONIZED LEGACY NEGATIVE COMPARISON")
    print("=" * 88)
    for name, metrics in (
        ("original_baseline_all_rows", output["original_baseline_all_rows"]),
        ("corrected_baseline_same_rows", baseline_metrics),
        ("harmonized_candidate", candidate_metrics),
    ):
        print(
            f"{name}: PR-AUC={metrics['pr_auc']:.4f}; AUROC={metrics['auroc']:.4f}; "
            f"recall@90spec={metrics['recall_at_90_spec']:.3f}; "
            f"IRIDIA AUROC={metrics['iridia_auroc']:.4f}; "
            f"legacy AUROC={metrics['legacy_auroc']:.4f}"
        )
    print("corrected_validation_windows:", output["corrected_validation_windows"])
    print("excluded_validation_post_af_windows:", output["excluded_validation_post_af_windows"])
    print("\nPROGRESSION GATES")
    for key, value in gates.items():
        print(f"{key}: {value}")
    print("passes_progression_gate:", output["passes_progression_gate"])
    print("outer_test_unlock: False")
    print("decision:", output["decision"])
    print("No outer-test predictions were evaluated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Apply the predeclared validation progression gate to two controlled repairs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from afib_unified.manifest import sha256_file
from afib_unified.validation_metrics import attach_manifest_metadata, summarize_predictions


CANDIDATES = ("groupnorm", "groupnorm-curriculum")


def zero_recall_fraction(diagnostics: dict, source: str) -> float:
    patient = diagnostics["by_source"][source]["patient_macro_at_source_threshold"]
    return patient["zero_recall_patients"] / patient["positive_patients"]


def progression_gates(candidate: dict, baseline: dict) -> dict:
    candidate_pr = candidate["ranking"]["pr_auc"]
    baseline_pr = baseline["ranking"]["pr_auc"]
    operating = candidate["global_operating_point"]
    return {
        "overall_pr_auc_at_least_20_percent_relative_gain": candidate_pr >= 1.20 * baseline_pr,
        "recall_at_least_0_40_at_90_percent_specificity": (
            operating["specificity"] >= 0.90 and operating["recall"] >= 0.40
        ),
        "legacy_auroc_at_least_0_60": (
            candidate["by_source"]["legacy"]["ranking"]["auroc"] >= 0.60
        ),
        "iridia_auroc_at_least_0_64": (
            candidate["by_source"]["iridia"]["ranking"]["auroc"] >= 0.64
        ),
        "iridia_zero_recall_patient_fraction_reduced": (
            zero_recall_fraction(candidate, "iridia")
            < zero_recall_fraction(baseline, "iridia")
        ),
        "legacy_zero_recall_patient_fraction_reduced": (
            zero_recall_fraction(candidate, "legacy")
            < zero_recall_fraction(baseline, "legacy")
        ),
    }


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
        "iridia_zero_recall_patient_fraction": zero_recall_fraction(diagnostics, "iridia"),
        "legacy_zero_recall_patient_fraction": zero_recall_fraction(diagnostics, "legacy"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-predictions", type=Path, required=True)
    parser.add_argument("--repairs-root", type=Path, required=True)
    parser.add_argument("--protocol-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest_path = args.protocol_dir / "unified_cached_manifest.csv"
    manifest = pd.read_csv(manifest_path)
    protocol = json.loads((args.protocol_dir / "protocol_summary.json").read_text())
    if sha256_file(manifest_path) != protocol["manifest_sha256"]:
        raise ValueError("manifest hash mismatch")
    baseline_predictions = attach_manifest_metadata(
        pd.read_csv(args.baseline_predictions), manifest
    )
    baseline = summarize_predictions(baseline_predictions)
    results = {
        "baseline-batchnorm": {
            "metrics": compact(baseline),
            "diagnostics": baseline,
        }
    }
    eligible = []
    for recipe in CANDIDATES:
        directory = args.repairs_root / recipe.replace("-", "_")
        summary_path = directory / "controlled_repair_summary.json"
        prediction_path = directory / "best_validation_predictions.csv"
        if not summary_path.is_file() or not prediction_path.is_file():
            raise FileNotFoundError(f"incomplete candidate output: {directory}")
        summary = json.loads(summary_path.read_text())
        if not summary.get("all_gates_pass"):
            raise RuntimeError(f"candidate integrity gates failed: {recipe}")
        if summary["manifest_sha256"] != protocol["manifest_sha256"]:
            raise ValueError(f"candidate manifest hash mismatch: {recipe}")
        predictions = attach_manifest_metadata(pd.read_csv(prediction_path), manifest)
        diagnostics = summarize_predictions(predictions)
        gates = progression_gates(diagnostics, baseline)
        passed = all(gates.values())
        results[recipe] = {
            "best_epoch": summary["best_epoch"],
            "metrics": compact(diagnostics),
            "progression_gates": gates,
            "passes_progression_gate": passed,
            "diagnostics": diagnostics,
        }
        if passed:
            eligible.append(recipe)
    winner = max(
        eligible,
        key=lambda name: results[name]["metrics"]["pr_auc"],
        default=None,
    )
    comparison = {
        "format": "afib_controlled_validation_repair_comparison_v1",
        "baseline_pr_auc": baseline["ranking"]["pr_auc"],
        "required_pr_auc": 1.20 * baseline["ranking"]["pr_auc"],
        "results": results,
        "selected_candidate": winner,
        "outer_test_unlock": winner is not None,
        "decision": (
            "progression gate passed; selected candidate may advance to locked outer evaluation"
            if winner is not None
            else "no candidate passed every progression gate; outer evaluation remains locked"
        ),
        "outer_test_evaluated": False,
        "manifest_sha256": protocol["manifest_sha256"],
        "splits_sha256": protocol["splits_sha256"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(comparison, indent=2, sort_keys=True))
    print("CONTROLLED REPAIR COMPARISON")
    print("=" * 72)
    print(f"Required PR-AUC: {comparison['required_pr_auc']:.4f}")
    for name, result in results.items():
        metrics = result["metrics"]
        print(
            f"{name}: PR-AUC={metrics['pr_auc']:.4f}; AUROC={metrics['auroc']:.4f}; "
            f"recall@90spec={metrics['recall_at_90_spec']:.3f}; "
            f"IRIDIA AUROC={metrics['iridia_auroc']:.4f}; "
            f"legacy AUROC={metrics['legacy_auroc']:.4f}"
        )
        if "progression_gates" in result:
            for gate, passed in result["progression_gates"].items():
                print(f"  {gate}: {passed}")
    print("selected_candidate:", winner)
    print("outer_test_unlock:", comparison["outer_test_unlock"])
    print("decision:", comparison["decision"])
    print("No outer-test patients were evaluated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

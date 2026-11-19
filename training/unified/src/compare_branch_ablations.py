"""Compare independently trained full, ECG-only and HRV-only validation models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score

from afib_unified.manifest import sha256_file
from afib_unified.validation_metrics import attach_manifest_metadata, summarize_predictions


EXPERIMENTS = {
    "full_fusion": "groupnorm_curriculum",
    "ecg_only": "groupnorm_curriculum_ecg_only",
    "hrv_only": "groupnorm_curriculum_hrv_only",
}


def compact(diagnostics: dict, predictions: pd.DataFrame) -> dict:
    source_labels = predictions.source_family.astype(str).eq("iridia").astype(int)
    source_auc = roc_auc_score(source_labels, predictions.probability)
    negative = predictions.label.eq(0)
    negative_source_auc = roc_auc_score(
        source_labels[negative], predictions.loc[negative, "probability"]
    )
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
        "source_score_separability_auc": max(source_auc, 1.0 - source_auc),
        "negative_source_score_separability_auc": max(
            negative_source_auc, 1.0 - negative_source_auc
        ),
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
    for name, folder in EXPERIMENTS.items():
        directory = args.runs_root / folder
        summary_path = directory / "controlled_repair_summary.json"
        prediction_path = directory / "best_validation_predictions.csv"
        if not summary_path.is_file() or not prediction_path.is_file():
            raise FileNotFoundError(f"incomplete branch experiment: {directory}")
        summary = json.loads(summary_path.read_text())
        if not summary.get("all_gates_pass"):
            raise RuntimeError(f"branch integrity gates failed: {name}")
        if summary["manifest_sha256"] != protocol["manifest_sha256"]:
            raise ValueError(f"branch manifest hash mismatch: {name}")
        predictions = attach_manifest_metadata(pd.read_csv(prediction_path), manifest)
        diagnostics = summarize_predictions(predictions)
        results[name] = {
            "best_epoch": summary["best_epoch"],
            "input_mode": summary["recipe"].get("input_mode", "full"),
            "metrics": compact(diagnostics, predictions),
        }
    full_pr = results["full_fusion"]["metrics"]["pr_auc"]
    ecg_pr = results["ecg_only"]["metrics"]["pr_auc"]
    hrv_metrics = results["hrv_only"]["metrics"]
    findings = {
        "ecg_only_retains_90_percent_of_full_pr_auc": ecg_pr >= 0.90 * full_pr,
        "hrv_only_has_positive_legacy_ranking": hrv_metrics["legacy_auroc"] > 0.50,
        "hrv_only_has_useful_legacy_ranking": hrv_metrics["legacy_auroc"] >= 0.60,
        "hrv_only_has_useful_iridia_ranking": hrv_metrics["iridia_auroc"] >= 0.64,
    }
    output = {
        "format": "afib_independent_branch_ablation_v1",
        "results": results,
        "findings": findings,
        "outer_test_unlock": False,
        "outer_test_evaluated": False,
        "decision_scope": (
            "diagnostic ablation only; use results to choose the next architecture, "
            "never to unlock outer evaluation"
        ),
        "manifest_sha256": protocol["manifest_sha256"],
        "splits_sha256": protocol["splits_sha256"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True))
    print("INDEPENDENT BRANCH ABLATION")
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
    print("\nFINDINGS")
    for key, value in findings.items():
        print(f"{key}: {value}")
    print("outer_test_unlock: False")
    print("No outer-test patients were evaluated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--forecast-run", required=True, type=Path)
    parser.add_argument("--protocol-dir", required=True, type=Path)
    parser.add_argument("--harmonized-manifest", required=True, type=Path)
    parser.add_argument("--code-archive", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    required = {
        "forecast_checkpoint.pt": args.forecast_run / "best_checkpoint.pt",
        "forecast_validation_predictions.csv": args.forecast_run / "best_validation_predictions.csv",
        "forecast_summary.json": args.forecast_run / "controlled_repair_summary.json",
        "unified_cached_manifest.csv": args.protocol_dir / "unified_cached_manifest.csv",
        "locked_patient_splits.json": args.protocol_dir / "locked_patient_splits.json",
        "protocol_summary.json": args.protocol_dir / "protocol_summary.json",
        "harmonized_cached_manifest.csv": args.harmonized_manifest,
        args.code_archive.name: args.code_archive,
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing freeze inputs:\n" + "\n".join(missing))

    summary = json.loads(required["forecast_summary.json"].read_text(encoding="utf-8"))
    if bool(summary.get("outer_test_evaluated", False)):
        raise ValueError("Refusing to freeze a development run that accessed outer test data")
    if int(summary.get("parameter_count", 0)) >= 250_000:
        raise ValueError("Forecast model exceeds the 250k parameter contract")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    copied = {}
    for frozen_name, source in required.items():
        destination = args.output_dir / frozen_name
        shutil.copy2(source, destination)
        copied[frozen_name] = {
            "source": str(source),
            "bytes": destination.stat().st_size,
            "sha256": sha256(destination),
        }

    decision = {
        "format": "afib_forecast_freeze_v1",
        "frozen_utc": datetime.now(UTC).isoformat(),
        "model_role": "20_minute_af_onset_risk_forecaster",
        "selected_recipe": "groupnorm_curriculum_corrected_evaluation",
        "parameter_budget": 250_000,
        "parameter_count": int(summary["parameter_count"]),
        "development_fold": int(summary.get("fold", 0)),
        "outer_test_evaluated_during_selection": False,
        "contract": {
            "sample_rate_hz": 250,
            "observation_minutes": 10,
            "segments": 20,
            "seconds_per_segment": 30,
            "forecast_horizon_minutes": 20,
            "ecg_shape": ["B", 20, 1, 7500],
            "hrv_shape": ["B", 20, 6],
        },
        "prohibited_changes": [
            "architecture", "preprocessing", "eligibility policy", "loss",
            "sampler", "selection metric", "threshold rule", "patient splits",
        ],
        "files": copied,
        "all_gates_pass": True,
    }
    freeze_path = args.output_dir / "FORECAST_FREEZE.json"
    freeze_path.write_text(json.dumps(decision, indent=2, sort_keys=True), encoding="utf-8")
    (args.output_dir / "LOCKED").write_text(sha256(freeze_path) + "\n", encoding="utf-8")
    print(json.dumps(decision, indent=2))
    print("\nFORECAST MODEL RECIPE AND DEVELOPMENT EVIDENCE FROZEN")


if __name__ == "__main__":
    main()

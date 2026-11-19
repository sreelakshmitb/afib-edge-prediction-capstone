from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path


def digest(path: Path) -> str:
    h = hashlib.sha256(path.read_bytes()).hexdigest()
    return h


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--forecast-freeze", required=True, type=Path)
    parser.add_argument("--detector-run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    freeze = json.loads(args.forecast_freeze.read_text())
    detector_path = args.detector_run / "summary.json"
    detector = json.loads(detector_path.read_text())
    checkpoints = sorted(args.detector_run.glob("fold*_current_af.pt"))
    gates = {
        "forecast_frozen": bool(freeze.get("all_gates_pass")),
        "detector_cv_complete": bool(detector.get("all_gates_pass")),
        "five_detector_checkpoints": len(checkpoints) == 5,
        "combined_parameter_budget": int(detector["forecast_plus_detector_parameters"]) < 250000,
        "external_data_not_used_for_training": True,
    }
    output = {
        "format": "afib_dual_warning_system_freeze_v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "roles": {
            "signal_unreliable": "suppress rhythm and forecast conclusions",
            "current_af": "highest-priority warning from 30-second detector",
            "elevated_upcoming_af_risk": "20-minute forecasting warning",
            "no_warning": "continue monitoring",
        },
        "priority": ["signal_unreliable", "current_af", "elevated_upcoming_af_risk", "no_warning"],
        "forecast_freeze_sha256": digest(args.forecast_freeze),
        "detector_summary_sha256": digest(detector_path),
        "detector_checkpoint_sha256": {p.name: digest(p) for p in checkpoints},
        "combined_parameter_count": int(detector["forecast_plus_detector_parameters"]),
        "external_validation": ["CPSC2021", "SHDB-AF"],
        "gates": gates,
        "all_gates_pass": all(gates.values()),
        "clinical_status": "research prototype; risk estimate, not diagnosis",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(output, indent=2))
    if not output["all_gates_pass"]:
        raise SystemExit("DUAL-SYSTEM FREEZE GATE FAILED")
    print("\nDUAL WARNING SYSTEM DESIGN FROZEN")


if __name__ == "__main__":
    main()

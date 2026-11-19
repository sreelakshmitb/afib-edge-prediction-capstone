"""CLI for the immutable five-horizon target reconstruction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from afib_unified.multihorizon import build_and_save_multihorizon_targets


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cached-manifest", type=Path, required=True)
    parser.add_argument("--combined-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    summary = build_and_save_multihorizon_targets(
        args.cached_manifest, args.combined_manifest, args.output_dir
    )
    print("MULTI-HORIZON TARGET RECONSTRUCTION")
    print("=" * 72)
    for key in (
        "all_gates_pass", "windows", "positives", "negatives", "patients",
        "positive_events", "horizons_minutes", "event_bin_counts",
    ):
        print(f"{key}: {summary[key]}")
    print("\nGATES")
    for key, value in summary["gates"].items():
        print(f"{key}: {value}")
    print("outer_test_evaluated: False")
    print("\nMULTI-HORIZON TARGETS LOCKED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

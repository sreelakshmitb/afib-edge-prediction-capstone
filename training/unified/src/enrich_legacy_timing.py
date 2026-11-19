"""Build the cache-preserving legacy timing correction manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from afib_unified.legacy_timing import build_and_save_harmonized_manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-dir", type=Path, required=True)
    parser.add_argument("--protocol-dir", type=Path, required=True)
    parser.add_argument("--afdb-root", type=Path, required=True)
    parser.add_argument("--ltafdb-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    summary = build_and_save_harmonized_manifest(
        target_dir=args.target_dir,
        protocol_dir=args.protocol_dir,
        afdb_root=args.afdb_root,
        ltafdb_root=args.ltafdb_root,
        output_dir=args.output_dir,
    )
    print("HARMONIZED LEGACY TIMING MANIFEST")
    print("=" * 76)
    for key in (
        "all_gates_pass", "windows", "training_eligible_windows",
        "evaluation_eligible_windows", "records_audited",
        "legacy_post_af_recovery_excluded", "legacy_eligible_strata",
        "outer_test_evaluated",
    ):
        print(f"{key}: {summary[key]}")
    print("\nGATES")
    for key, value in summary["gates"].items():
        print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

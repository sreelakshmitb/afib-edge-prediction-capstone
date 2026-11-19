import argparse
import json
from pathlib import Path

from afib_unified.manifest import build_and_save


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--combined-manifest", type=Path, required=True)
    parser.add_argument("--iridia-plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    summary = build_and_save(
        args.combined_manifest, args.iridia_plan, args.output_dir, args.seed
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

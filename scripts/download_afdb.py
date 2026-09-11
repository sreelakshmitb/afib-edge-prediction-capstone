"""Download one MIT-BIH AFDB record and rhythm annotations outside Git."""

from __future__ import annotations

import argparse
from pathlib import Path

from data.afdb import download_afdb_record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--record", default="04015")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    download_afdb_record(args.raw_dir, args.record, overwrite=args.overwrite)
    print("AFDB_DOWNLOAD=PASS")
    print(f"record={args.record}")
    print(f"raw_dir={args.raw_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

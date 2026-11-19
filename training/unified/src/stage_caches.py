"""Resume-safe staging of only training cache files from Drive to local SSD."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path

import pandas as pd

from afib_unified.manifest import sha256_file


FILES = ("ecg.npy", "hrv.npy", "metadata.json", "COMPLETE")


def copy_verified(source: Path, target: Path) -> int:
    if target.is_file() and target.stat().st_size == source.stat().st_size:
        return 0
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".partial")
    if temporary.exists():
        temporary.unlink()
    shutil.copy2(source, temporary)
    if temporary.stat().st_size != source.stat().st_size:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"staged size mismatch: {source}")
    os.replace(temporary, target)
    return source.stat().st_size


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy-source", type=Path, required=True)
    parser.add_argument("--iridia-source", type=Path, required=True)
    parser.add_argument("--protocol-dir", type=Path, required=True)
    parser.add_argument("--stage-root", type=Path, required=True)
    args = parser.parse_args()
    manifest_path = args.protocol_dir / "unified_cached_manifest.csv"
    manifest = pd.read_csv(manifest_path)
    records = manifest[["cache_family", "cache_record_id"]].drop_duplicates()
    sources = {"legacy": args.legacy_source, "iridia": args.iridia_source}
    required_bytes = 0
    tasks = []
    for row in records.itertuples(index=False):
        family, record = str(row.cache_family), str(row.cache_record_id)
        source_dir = sources[family] / (record.replace(":", "__") if family == "legacy" else record)
        target_dir = args.stage_root / family / source_dir.name
        for filename in FILES:
            source = source_dir / filename
            if not source.is_file():
                raise FileNotFoundError(source)
            required_bytes += source.stat().st_size
            tasks.append((source, target_dir / filename))
    args.stage_root.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(args.stage_root).free
    existing = sum(
        source.stat().st_size
        for source, target in tasks
        if target.is_file() and target.stat().st_size == source.stat().st_size
    )
    remaining = max(0, required_bytes - existing)
    if free < remaining + 1024**3:
        raise RuntimeError(
            f"insufficient local storage: need {remaining / 1024**3:.2f} GiB plus 1 GiB, "
            f"have {free / 1024**3:.2f} GiB"
        )
    copied = 0
    started = time.time()
    for index, (source, target) in enumerate(tasks, start=1):
        copied += copy_verified(source, target)
        if index == 1 or index % 100 == 0 or index == len(tasks):
            print(
                f"STAGE {index}/{len(tasks)}; newly_copied={copied / 1024**3:.2f} GiB; "
                f"elapsed={(time.time() - started) / 60:.1f} min",
                flush=True,
            )
    summary = {
        "format": "afib_unified_local_stage_v1",
        "records": int(len(records)),
        "files": len(tasks),
        "required_bytes": required_bytes,
        "newly_copied_bytes": copied,
        "manifest_sha256": sha256_file(manifest_path),
        "legacy_stage": str(args.stage_root / "legacy"),
        "iridia_stage": str(args.stage_root / "iridia"),
        "all_gates_pass": all(target.is_file() and target.stat().st_size == source.stat().st_size for source, target in tasks),
    }
    (args.stage_root / "STAGE_COMPLETE.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not summary["all_gates_pass"]:
        raise RuntimeError("local stage gate failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

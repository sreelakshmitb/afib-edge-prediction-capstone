"""Run one real AFDB record through the preprocessing/window/HRV pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from data.contracts import HRV_FEATURE_ORDER
from data.pipeline import build_one_afdb_observation, save_observation_npz


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--record", default="04015")
    parser.add_argument("--lead-index", type=int, default=0)
    parser.add_argument("--stride-seconds", type=int, default=480)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/afdb_smoke/afdb_window.npz"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    window, af_interval_count = build_one_afdb_observation(
        args.raw_dir,
        args.record,
        lead_index=args.lead_index,
        stride_seconds=args.stride_seconds,
        prefer_positive=True,
    )
    save_observation_npz(window, args.output)

    missing_by_feature = np.count_nonzero(~window.hrv_valid_mask, axis=0).tolist()
    print("AFDB_SMOKE_TEST=PASS")
    print(f"database={window.database}")
    print(f"record_id={window.record_id}")
    print(f"patient_id={window.patient_id}")
    print(f"original_fs_hz={window.original_fs_hz}")
    print(f"model_fs_hz={window.model_fs_hz}")
    print(f"lead_index={window.lead_index}")
    print(f"lead_name={window.lead_name}")
    print(f"af_interval_count={af_interval_count}")
    print(
        f"observation_seconds={window.observation_start_seconds:.3f}:"
        f"{window.observation_end_seconds:.3f}"
    )
    print(
        f"prediction_seconds={window.prediction_start_seconds:.3f}:"
        f"{window.prediction_end_seconds:.3f}"
    )
    print(f"ecg_shape={window.ecg.shape}")
    print(f"hrv_shape={window.hrv.shape}")
    print(f"label={window.label}")
    print(f"hrv_feature_order={list(HRV_FEATURE_ORDER)}")
    print(f"hrv_missing_by_feature={missing_by_feature}")
    print(f"artifact={args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# AFib edge prediction capstone

**AI-Based Single-Lead ECG Monitoring System for Early Atrial Fibrillation Prediction Using Edge Computing**

A research prototype that uses ten minutes of single-lead ECG to estimate AF onset risk within the following twenty minutes. MAX30003 acquisition and ESP32 Wi-Fi transport feed a Raspberry Pi 4B, which runs preprocessing and the frozen ONNX forecaster and serves an Android application.

## Final project

| Item | Published implementation |
| --- | --- |
| Model | SE-CNN-HRV-UniLSTM, GroupNorm/curriculum, 159,503 parameters |
| Frozen checkpoint | [forecast_checkpoint.pt](artifacts/frozen_forecast_v1/forecast_checkpoint.pt) |
| Pi graph and preprocessing | [deployment/pi/model](deployment/pi/model) |
| Model source and recipe | [training/unified](training/unified), `build_model("proposed-groupnorm")` |
| Model input | 250 Hz; ECG `[B,20,1,7500]`; HRV `[B,20,6]` |
| Output | One logit; sigmoid score; validation-selected cutoff **0.3310546875** |
| Pi receiver, inference and app bridge | [deployment/pi](deployment/pi) |
| Android | [android](android), imported v1.4.8 live-resume source |
| ESP32 | [live Wi-Fi firmware](firmware/afib_live_wifi/afib_live_wifi.ino) |
| PCB | [Rev B KiCad, fabrication and mechanical files](hardware/pcb) |
| Integrity | [SHA-256 release manifest](artifacts/RELEASE_MANIFEST.json) |

## Results for the frozen model

The once-only patient-disjoint outer fold-0 test contains **1,809 windows**, including 236 positives. AP is **0.22175**, AUROC **0.66424**, recall **23.31%**, and specificity **90.97%** at the frozen threshold. Confusion counts: TP 55, FN 181, FP 142, TN 1,431. It detected 38/107 eligible events, including 9/48 events in the 16–20 minute lead band.

These are modest research results with limited sensitivity. A below-threshold score does not rule out AF. This is a research prototype, not a validated diagnostic device.

The later fivefold BCE development assessment uses different checkpoints. See [results and limitations](docs/RESULTS.md), [raw test summary](results/frozen_forecast_sealed_test_v1/summary.json), and [predictions](results/frozen_forecast_sealed_test_v1/test_predictions.csv).

## Start here

1. [Pi setup and live/replay commands](deployment/pi/README.md)
2. [Model card and loading](docs/FINAL_MODEL.md)
3. [Datasets and experiment scope](docs/DATASETS.md)
4. [Training and notebooks](training/README.md)
5. [Hardware and firmware](hardware/README.md)
6. [Current state and remaining evidence](PROJECT_STATE.md)
7. [Import provenance and cleanup](docs/RELEASE_NOTES_2026-11-20.md)

Live mode collects a real ten minutes before its first forecast. Replay transfers a prerecorded ten-minute waveform faster than real time. Presentation simulation is a UI demonstration and must be off for actual Pi inference.

## Verification

```bash
python scripts/verify_release.py
python -m pip install -r deployment/pi/requirements.txt
python scripts/verify_frozen_model.py
```

The first command verifies published hashes without ML dependencies. The model check executes ONNX on three supplied validation fixtures. Add `--checkpoint` after installing PyTorch to compare both runtimes.

## Project team and attribution

Joint work of [Sreelakshmi T B](https://github.com/sreelakshmitb), [Jagadjith U](https://github.com/Jagadjith-U), and [Aravind V S](https://github.com/aravindd3). Research, software, experiments, analysis and documentation were collaborative; commit authorship records who uploaded a change.

The CNN-HRV-UniLSTM direction is based on the Zhang et al. reference documented in [the reproducibility audit](docs/PAPER_REPRODUCIBILITY_AUDIT.md). SE attention, training repairs, eligibility work and device integration are capstone work; no superiority from SE alone is claimed.

The earlier `afib/`, `src/`, `configs/`, tests and `model_bundles/se_last_mask_seed7/` remain historical development material. Their checkpoints and preprocessing are not interchangeable with the frozen deployment. No complete raw dataset is included; deployment includes three small validation fixtures from the supplied export.

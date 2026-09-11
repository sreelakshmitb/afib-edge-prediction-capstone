# AFib Edge Prediction Capstone — Project State

## Current Phase
Phase 1 — environment, repository, datasets, and preprocessing.

## Primary Technical Source
“A Lightweight Deep Learning Model for Short-Term Atrial Fibrillation Prediction from Single-Lead ECG Signals in Healthcare 4.0” — Zhang et al., ICASSP 2026.

## Core Objective
Predict AF onset within the subsequent 20 minutes using sinus-rhythm single-lead ECG.

## Fixed Tensor Contracts
- ECG: [B,20,1,7500]
- HRV: [B,20,6]

## Fixed HRV Order
1. RMSSD
2. SDNN
3. LF/HF
4. Mean RR
5. pNN50
6. Sample Entropy

## Hard Constraints
- ECG model sampling rate: 250 Hz
- 10-minute observation window
- 20 chronological 30-second segments
- Patient-wise GroupKFold(n_splits=5)
- Group-aware validation inside each outer-training fold
- No patient leakage
- Model parameters <250,000
- Raspberry Pi 4B, Cortex-A72, 2GB RAM
- ONNX Runtime CPU deployment
- Target model-only latency <50 ms, subject to physical measurement
- Non-Raspberry-Pi BOM <₹6,000

## Primary Proposed Model
SE-CNN-HRV-UniLSTM

## Reference Model
Paper-aligned CNN-HRV-UniLSTM (~189k parameters reported by the paper; exact reproduction architecture details remain to be resolved where omitted).

## Verified Repository Status
- Repository bootstrap/documentation files exist on `main`.
- Phase 1 source tree and Python dependency manifest are being added in `feature/phase1-bootstrap`.
- A fixed-contract module and unit tests are being added for sampling rate, window geometry, HRV ordering, GroupKFold count, and parameter ceiling.
- A lightweight GitHub Actions workflow is being added for source compilation and contract tests.
- `.gitignore` is being hardened against raw ECG data, generated arrays, checkpoints, ONNX files, and experiment logs.
- No AFDB/LTAFDB raw data is committed.
- No dataset acquisition/loader implementation exists yet.
- No preprocessing outputs, manifests, windows, HRV arrays, checkpoints, trained models, or experiment results have been verified in GitHub.

## Phase 1 Repository Scaffold
Expected tracked structure after the bootstrap PR:
- `configs/`
- `src/data/`
- `src/models/`
- `src/training/`
- `src/deployment/`
- `scripts/`
- `tests/`
- `artifacts/` (placeholder only; generated artifacts ignored)

## Still Missing
- AFDB acquisition/loading code
- ECG resampling and 0.5–40 Hz filtering
- AF rhythm annotation parsing and causal onset labeling
- R-peak detection and RR interval extraction
- Six-feature HRV extraction with explicit short-window edge-case handling
- Generated-window integrity tests
- Group-aware outer/inner split implementation and patient-ID audits
- Dataset/DataLoader implementation
- Paper-reference CNN-HRV-UniLSTM
- Proposed SE-CNN-HRV-UniLSTM
- Smoke training and real training
- Metrics/checkpoints and ONNX export
- Physical Raspberry Pi integration and benchmarking

## Validation State
Repository-level contract tests are defined in the bootstrap branch. Full dependency installation and later data/model tests remain to be executed; no training success is claimed.

## Next Task
After the Phase 1 bootstrap is merged, implement AFDB acquisition/loading first, keeping raw PhysioNet data outside Git.

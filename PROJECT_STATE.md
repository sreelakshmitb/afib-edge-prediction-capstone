# AFib Edge Prediction Capstone — Project State

## Current Phase
Real AFDB preprocessing and verification; candidate training infrastructure implemented.

## Team
- Sreelakshmi — sreelakshmitb
- Jagadjith — Jagadjith-U
- Aravind — aravindd3

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
- No patient leakage
- Model parameters <250,000
- Raspberry Pi 4B, Cortex-A72, 2GB RAM
- ONNX Runtime CPU deployment
- Target model-only latency <50 ms, subject to physical measurement
- Non-Raspberry-Pi BOM <₹6,000

## Primary Proposed Model
SE-CNN-HRV-UniLSTM — CAPSTONE EXTENSION, 198,425 parameters.
The final deployed architecture is undecided.

## Reference Model
CNN-HRV-UniLSTM — PAPER REPRODUCTION / REFERENCE BASELINE, 187,393 parameters.
This architecture is attributed to Zhang et al., not an original capstone contribution.
Unspecified details are marked REPRODUCTION CHOICE in the paper audit.

## Current Status
- GitHub repository created
- Team repository setup in progress
- Primary paper available
- University thesis template available
- Current capstone circular available
- Local branch: codex/afdb-pipeline; implementation is committed in focused stages.
- Python 3.14.6 / CPU PyTorch 2.14.0 environment verified; dependencies pinned.
- WFDB loading, verified/resumable AFDB downloads, causal ECG preprocessing,
  six HRV features, labeling and independent manifest audit implemented.
- Real record 04015 smoke test passed: 54 observations (3 positive, 51 negative).
- On resuming the interrupted run, nine completed records contained 314 observations
  (50 positive, 264 negative); the remaining 14 records are being processed.
- Dataset/DataLoader, outer GroupKFold(5), group-aware inner holdout, training-only
  HRV transform, both candidate models and checkpoint training loop implemented.
- 15 tests passed before resuming changes. A real two-example forward/backward
  code smoke test had finite loss and gradients; this is not a validation result.
- No trained checkpoint or real-fold performance is yet available.
- Raw data, arrays and generated manifests: C:/Users/Lenovo/afib-edge-data (outside Git).
- FP32 export code exists; it has not yet exported a trained checkpoint.

## Unresolved
- Full AFDB manifest audit and patient-separated real training
- Final architecture selection without using outer-test results
- Full-paper reproduction gaps, documented in docs/PAPER_REPRODUCIBILITY_AUDIT.md
- Exact hardware breakout boards
- Actual Raspberry Pi latency
- Final experimental results

## Next Task
Finish remaining AFDB records, assemble and audit the complete manifest, then run
patient-separated smoke training and first real folds for the two candidates.

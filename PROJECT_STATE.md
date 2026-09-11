# AFib Edge Prediction Capstone — Project State

## Current Phase
Both candidates completed all five patient-separated outer folds; results and experimental FP32 exports verified.

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
- All 23 signal-bearing AFDB records processed; 20 eligible patients yielded
  665 observations (92 positive, 573 negative). Three records yielded none.
- Dataset/DataLoader, outer GroupKFold(5), group-aware inner holdout, training-only
  HRV transform, both candidate models and checkpoint training loop implemented.
- 17 tests passed on the final resume audit. Both smoke runs and all ten full
  candidate/fold runs have saved checkpoints. No completed fold was restarted.
- Fixed recipe: maximum 20 epochs, patience 5, AdamW lr 5e-4, batch size 8;
  checkpoint selection by minimum inner-validation BCE, fixed threshold 0.5.
- Five-fold mean AUROC: reference 0.595085 ± 0.073127; SE 0.578832 ± 0.054145
  (sample standard deviation). Recall is low; these are experimental models.
- All fold metrics, confusion matrices, selected epochs, verification details and
  artifact fingerprints are in docs/EXPERIMENT_RESULTS_20260911.md.
- Raw data, arrays and generated manifests: C:/Users/Lenovo/afib-edge-data (outside Git).
- Both fold-0 FP32 ONNX exports exist; saved parity checks passed for batches 1
  and 2. Maximum absolute errors: reference 5.96e-8, SE 1.19e-7.

## Unresolved
- Final architecture selection without using outer-test results
- Full-paper reproduction gaps, documented in docs/PAPER_REPRODUCIBILITY_AUDIT.md
- Exact hardware breakout boards
- Actual Raspberry Pi latency
- Independent validation and improved sensitivity before any deployment claim

## Next Task
The fixed first experiment is complete. Preserve its artifacts and recipe.
Any follow-up experiment needs a separately documented protocol; do not tune
architecture or thresholds using these already-observed outer-test results.

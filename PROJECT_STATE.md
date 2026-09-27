# AFib Edge Prediction Capstone — Project State

## Current Phase
The team's current supplied project is the `se_last_mask`, seed-7 bundle in
`model_bundles/se_last_mask_seed7`. See README.md and docs/CURRENT_BUNDLE_IMPORT.md.
It contains five development inner-validation checkpoints and their evidence.
The older local pipeline and the historical status below are preserved; no
training or outer-test evaluation was resumed for this upload.

## Team
- Sreelakshmi — sreelakshmitb
- Jagadjith — Jagadjith-U
- Aravind — aravindd3

## Primary Technical Source
“A Lightweight Deep Learning Model for Short-Term Atrial Fibrillation Prediction from Single-Lead ECG Signals in Healthcare 4.0” — Zhang et al., ICASSP 2026.

## Core Objective
Build a lightweight model that predicts AF onset within the subsequent 20 minutes
from single-lead ECG and generalizes to patients unseen during training. Exact
paper replication is not the objective. Annotation-eligible non-AF observations
are not guaranteed independently adjudicated pure sinus rhythm.

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
SE-CNN-HRV-UniLSTM — CAPSTONE EXTENSION, 198,425 parameters.
The final deployed architecture is undecided.

## Reference Model
CNN-HRV-UniLSTM — PAPER-BASED REFERENCE BASELINE, 187,393 parameters.
This architecture is attributed to Zhang et al., not an original capstone contribution.
Unspecified details are marked REPRODUCTION CHOICE in the paper audit.

## Current Status
- GitHub repository created
- Team repository setup in progress
- Primary paper available
- University thesis template available
- Current capstone circular available
- Local branch: codex/combined-cohort-development; PILOT_V1 report is preserved at 8c36169.
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
- Complete targeted inner-only development; no combined-cohort predictive result is claimed yet
- Public datasets lack a cross-dataset person linkage; namespaced record grouping
  and duplicate checks cannot prove distinct biological identities
- Final architecture selection without using outer-test results
- Full-paper reproduction gaps, documented in docs/PAPER_REPRODUCIBILITY_AUDIT.md
- Exact hardware breakout boards
- Actual Raspberry Pi latency
- Independent validation and improved sensitivity before any deployment claim

## Next Task
Continue the existing targeted-v1 development run under
docs/COMBINED_DEVELOPMENT_PROTOCOL.md; preserve completed candidates and resume
interrupted candidates from their saved state. The combined cohort contains
3,541 windows (278 positive, 3,263 negative) from 67 contributing patient groups.
All 109 source groups are accounted for, including 42 zero-window groups.
The cohort audit and fingerprints are summarized in docs/COMBINED_COHORT_AUDIT.md.
All 30 tests passed before resuming acquisition. Final outer evaluation remains
gated on completed inner selections. No new predictive performance is claimed.

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
Paper-aligned CNN-HRV-UniLSTM (~189k parameters reported by the paper; exact omitted architecture details remain reproduction choices).

## Verified Main Status
- Phase 1 repository bootstrap from PR #2 is merged on `main`.
- Minimal `configs/`, `src/`, `scripts/`, `tests/`, and ignored `artifacts/` structure exists.
- CI compiles Python sources and runs repository contract tests.
- Raw/local `data/` and generated model/data artifacts are ignored by Git.
- No raw ECG dataset, preprocessing output, checkpoint, trained model, or experiment result is committed.

## AFDB Data Pipeline Branch
`feature/afdb-data-pipeline` implements the first real-data path:
- configurable local AFDB raw-data directory
- one-record PhysioNet/WFDB download helper
- header/identity loading without reading the complete record
- WFDB `atr` rhythm parsing in original-record seconds
- explicit AF interval extraction
- causal 10-minute observation / 20-minute prediction labeling
- exclusion of observations not fully annotated as normal rhythm
- bounded one-window ECG loading for the smoke test
- resampling to 250 Hz only when required
- 0.5–40 Hz zero-phase Butterworth filtering
- WFDB XQRS R-peak detection as a documented reproduction choice
- six HRV features in the fixed project order
- NaN plus validity masks/reasons for unavailable HRV values
- [20,1,7500] ECG and [20,6] HRV smoke-test output
- local NPZ artifact output under ignored `artifacts/`

## Reproduction Choices Introduced in AFDB Pipeline
- For this smoke pipeline, AFDB `record_id` is retained as `patient_id` so no unverified subject identity is invented. Before patient-wise cross-validation, AFDB record-to-subject identity must be explicitly verified or mapped.
- AFDB lead index 0 is used for the smoke test because the paper does not identify the lead.
- WFDB XQRS is used because the paper states R-peak detection but does not name the detector.
- 20% observation overlap is interpreted as an 80% stride (480 seconds).
- Short-window LF/HF and Sample Entropy are left missing when minimum support/stability checks fail.

## Validation State
Synthetic repository tests cover resampling length, filtering shape/finite output, 30-second/20-segment geometry, annotation timing, HRV order and missingness, causal labels, ongoing-AF exclusion, and identity preservation.

A real AFDB record has NOT yet been claimed to pass the pipeline. The branch must remain unmerged until the local real-record smoke test succeeds and its console output is reviewed.

## Next Task
Run record `04015` locally through:
raw AFDB record -> WFDB rhythm annotations -> bounded ECG load -> preprocessing -> window generation -> R-peaks/RR -> HRV -> ECG/HRV/label output.

After successful local smoke verification, merge the AFDB data-pipeline PR and then implement patient-wise splitting / dataset loading.

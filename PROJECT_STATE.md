# AFib Edge Prediction Capstone — Project State

## Current Phase
Project setup and ground-truth audit.

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
SE-CNN-HRV-UniLSTM

## Reference Model
Paper-aligned CNN-HRV-UniLSTM

## Current Status
- GitHub repository created
- Team repository setup in progress
- Primary paper available
- University thesis template available
- Current capstone circular available
- Coding has not started

## Unresolved
- Exact reproduction assumptions from paper
- Exact dataset preprocessing details
- Exact six-feature HRV implementation decisions
- Exact hardware breakout boards
- Actual Raspberry Pi latency
- Final experimental results

## Next Task
Complete Project Ground Truth Register and Paper Reproducibility Audit before implementation.
# AFib Edge Prediction

Clean, leakage-safe implementation of the data and evaluation foundation for
20-minute-ahead atrial-fibrillation prediction from single-lead ECG.

## Fixed contract

- Common sampling rate: 250 Hz
- Observation: 10 minutes, represented as 20 contiguous 30-second segments
- ECG tensor: `[B, 20, 1, 7500]`
- HRV tensor: `[B, 20, 6]`
- HRV order: RMSSD, SDNN, LF/HF, Mean RR, pNN50, Sample Entropy
- Outer evaluation: `GroupKFold(n_splits=5)` grouped by patient
- Model limit: fewer than 250,000 trainable parameters
- External tests: CPSC2021 and SHDB-AF, without fine-tuning

The previous implementation and checkpoints are legacy evidence only. They are
not inputs to this pipeline.

## Current AP-only optimization experiment

V8 showed that equal record/patient weighting reduced validation AP by 0.0165,
while adding the fixed Smooth-AP term recovered 0.0104 AP relative to that
weighted objective. V9 therefore isolates the remaining high-value question:
does Smooth-AP improve the winning masked-BCE baseline when natural window
weighting is preserved?

The 20-minute target, 159,503-parameter model, data, folds, augmentation,
optimizer, seed, and epoch budget remain fixed. No outer-test windows are read.
See `COLAB_AP_ONLY_RUNBOOK.md` and `docs/AP_ONLY_V1.md`.

## Completed experiment: patient-equal and AP-aligned training

Patient-equal BCE and patient-equal BCE plus Smooth-AP did not pass the locked
five-fold promotion gate. See `COLAB_FINAL_OPTIMIZATION_RUNBOOK.md` and
`docs/FINAL_OPTIMIZATION_V1.md` for the historical experiment.

## Completed experiment: 20-minute horizon curriculum

The fixed 5→10→15→20-minute curriculum did not improve five-fold validation
AP or the within-record AUROC guardrail, so it is not part of the final run.
See `COLAB_HORIZON_CURRICULUM_RUNBOOK.md` and
`docs/HORIZON_CURRICULUM_V1.md` for the historical experiment.

## Completed experiment: AFPDB prediction-aligned pretraining

That experiment used only the labelled PAF-subject learning pairs from
the PhysioNet PAF Prediction Challenge Database. It builds the same 10-minute
ECG/HRV tensor contract as the target task, pretrains the complete sequence
representation for a fixed source budget without a source-validation claim, and compares transfer
against the existing masked SE baseline on identical development folds. See
`COLAB_AFPDB_RUNBOOK.md` and `docs/AFPDB_PREDICTION_V1.md`.

## Completed experiment: aligned Icentia11k forecast audit

The first Icentia experiment successfully learned 30-second normal-versus-AF
morphology, but direct encoder transfer did not improve the target forecast.
The current version therefore adds a resumable, annotation-only feasibility
audit for exact 10-minute normal observations followed by AF/AFL onset within
20 minutes. See `docs/ICENTIA_FORECAST_AUDIT_V1.md` and the first section of
`COLAB_RUNBOOK.md`. No additional ECG signals should be downloaded until that
audit passes its patient-level gate.

## Completed experiment: 30-second Icentia11k encoder pretraining

The v5 experiment keeps the 159,503-parameter SE-CNN-HRV-UniLSTM unchanged and
pretrains only its ECG segment encoder on a bounded, patient-disjoint subset of
Icentia11k (normal versus AF/AFL 30-second rhythm segments). The downstream
comparison isolates random versus pretrained initialization under the same
masking, loss, folds, seed, and epoch budget. See `COLAB_RUNBOOK.md` for the
resumable Colab commands and `docs/ICENTIA_PRETRAINING_V1.md` for the locked
scientific design.

## Phase 1 status

This checkpoint provides:

- numerically stable 0.5-40 Hz filtering and polyphase resampling;
- rhythm-state parsing that correctly closes intervals at rhythm changes and at
  the actual recording duration;
- window labeling that excludes AF-overlapping observations and right-censored
  windows without a complete 20-minute future horizon;
- the fixed six-feature HRV contract with missing values represented explicitly;
- five patient-wise outer folds with executable leakage assertions;
- synthetic unit tests for the critical invariants.

Dataset-specific ingestion and real manifests should be added only after the
actual AFDB/LTAFDB directory layout and annotation inventory have been audited.

The audit is now complete. AFDB contributes 23 usable signal records at 250 Hz
(the header-only records `00735` and `03665` are skipped), and LTAFDB contributes
84 two-lead records at 128 Hz. The primary manifest policy accepts only rhythm
`N` as annotated sinus rhythm; every other rhythm is excluded from observation
windows rather than silently treated as normal.

## Build the real AFDB + LTAFDB manifest in Colab

```bash
python -m afib_edge.wfdb_manifest \
  --afdb /content/drive/MyDrive/af_proj/init_model/data/afdb \
  --ltafdb /content/drive/MyDrive/af_proj/data/ltafdb \
  --afdb-metadata /content/wfdb_metadata/afdb \
  --ltafdb-metadata /content/wfdb_metadata/ltafdb \
  --output-dir /content/drive/MyDrive/AFib_Capstone/manifests/v1
```

The optional metadata paths avoid repeated reads through an unstable Drive
mount while preserving the true signal paths in the manifest. This reads
headers and `atr` annotations only. It creates `windows.csv`,
`folds.json`, and `summary.json`; it does not load or copy the large ECG signal
arrays. Review `summary.json` before signal preprocessing begins.

## Build the preprocessing cache

Always run a one-record smoke test first:

```bash
python -m afib_edge.preprocess_cache \
  --manifest /content/afib_manifest_v1/windows.csv \
  --output-dir /content/afib_cache_smoke \
  --stage-dir /content/wfdb_stage \
  --limit-records 1
```

The full resumable run is:

```bash
python -m afib_edge.preprocess_cache \
  --manifest /content/afib_manifest_v1/windows.csv \
  --output-dir /content/drive/MyDrive/AFib_Capstone/cache/v1 \
  --stage-dir /content/wfdb_stage \
  --resume
```

Each completed record has its own directory and `COMPLETE` marker. The signal is
read from Drive once, staged locally, filtered once, resampled once, and passed
through WFDB XQRS once. Arrays are written as exact-contract `.npy` files with
SHA-256 hashes. HRV failures remain NaN for training-fold-only imputation.
Sample Entropy uses the ordinary count-ratio estimator when defined and a
documented one-match finite-sample cap when a short 30-second RR sequence has
zero template matches; genuinely insufficient RR sequences still remain NaN.
R-peaks are detected with deterministic 30-minute XQRS cores and 15-second
overlap. This bounds pathological XQRS backsearch on noisy day-long Holter
records while retaining boundary context. Chunk progress is printed explicitly.

To benchmark one problematic record before a full run:

```bash
python -m afib_edge.preprocess_cache \
  --manifest /content/afib_manifest_v1/windows.csv \
  --output-dir /content/afib_cache_117_chunked \
  --stage-dir /content/wfdb_stage \
  --subject-id ltafdb:117
```

## Run tests

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

## Phase 2 models

The package now provides two paper-aligned models:

- `PaperCNNHRVUniLSTM`: three-layer 1-D CNN morphology encoder + six HRV
  features + one-layer unidirectional LSTM;
- `SECNNHRVUniLSTM`: the same reference model with squeeze-excitation gates in
  the CNN blocks.

Both consume `ecg=[B,20,1,7500]` and `hrv=[B,20,6]`, return one uncalibrated
logit per window, and remain below the 250,000 trainable-parameter limit. HRV
imputation is intentionally not performed inside the model: fit it on each
training fold and apply the resulting statistics to validation/test data.

Install the training dependency in Colab or the training environment:

```bash
pip install -e '.[train]'
```

Example construction:

```python
from afib_edge.models import build_model

model = build_model("proposed")
print(model.parameter_count)
logits = model(ecg, hrv)
```

## One-fold training smoke test

The lazy cache dataset opens record arrays as memory maps and returns one
window at a time. For each outer fold, HRV medians and standardization scales
are fitted only on the inner training patients; the same fitted values are
then applied to validation and outer-test windows.

Install the training dependency and run a bounded smoke test first:

```bash
python -m afib_edge.train \
  --cache-root /content/drive/MyDrive/AFib_Capstone/cache/v2_chunked_xqrs \
  --manifest /content/afib_manifest_v1/windows.csv \
  --model proposed \
  --fold 0 \
  --epochs 1 \
  --batch-size 2 \
  --max-train-batches 3 \
  --max-eval-batches 2 \
  --output-dir /content/drive/MyDrive/AFib_Capstone/runs/phase2_smoke
```

The smoke test should produce a checkpoint, a JSON summary, finite losses, and
patient-disjoint train/validation/test counts before any full training run is
started.

## Important evaluation rule

Do not balance, normalize, impute, select thresholds, calibrate probabilities,
or stop training using an outer-test fold. Fit every learned preprocessing
quantity using only the training portion of the corresponding outer fold.

For full runs, checkpoint selection uses inner-validation PR-AUC rather than a
fixed decision threshold. After loading the best checkpoint, the F1-maximizing
threshold is selected from inner-validation predictions only and then frozen
for outer-test metrics. AUROC, PR-AUC, confusion counts, metrics at the selected
threshold, and metrics at 0.5 are all retained.

An older checkpoint can be evaluated without retraining:

```bash
python -m afib_edge.evaluate_checkpoint \
  --cache-root /content/drive/MyDrive/AFib_Capstone/cache/v2_chunked_xqrs \
  --manifest /content/afib_manifest_v1/windows.csv \
  --checkpoint /content/drive/MyDrive/AFib_Capstone/runs/full_proposed_f0/proposed_fold0.pt \
  --device cuda \
  --output-dir /content/drive/MyDrive/AFib_Capstone/runs/full_proposed_f0/evaluation
```

This writes a JSON audit summary plus validation and outer-test prediction CSVs.

## Training recipe pilot

The cache audit showed substantial record-to-record ECG amplitude variation.
Training therefore exposes per-segment z-score normalization as an explicit
option. The imbalance recipe can also be switched between the paper-style
balanced sampler plus focal loss and a natural-shuffle BCE control. A
validation-only pilot compares these choices without iterating the outer-test
windows:

```bash
python -m afib_edge.pilot \
  --cache-root /content/drive/MyDrive/AFib_Capstone/cache/v2_chunked_xqrs \
  --manifest /content/afib_manifest_v1/windows.csv \
  --model proposed \
  --fold 0 \
  --epochs 12 \
  --patience 4 \
  --device cuda \
  --output-dir /content/drive/MyDrive/AFib_Capstone/runs/recipe_pilot_f0
```

The pilot ranks recipes by inner-validation PR-AUC. Only after selecting a
recipe should a full outer-test evaluation be run.

For the locked selected recipe, the resumable cross-validation runner evaluates
both architectures and writes one checkpoint and summary per fold:

```bash
python -m afib_edge.cross_validate \
  --cache-root /content/drive/MyDrive/AFib_Capstone/cache/v2_chunked_xqrs \
  --manifest /content/afib_manifest_v1/windows.csv \
  --models paper proposed \
  --folds 0 1 2 3 4 \
  --epochs 50 \
  --patience 10 \
  --device cuda \
  --ecg-normalization segment_zscore \
  --sampling natural \
  --loss bce \
  --output-dir /content/drive/MyDrive/AFib_Capstone/runs/cv_zscore_bce \
  --resume
```

If Colab disconnects, rerunning the same command resumes completed folds.

## Historical validation-only quality sprint v3

The v2 development screen found no material improvement from robust HRV
preprocessing or replacing BatchNorm with GroupNorm. Version 3 therefore runs
a pre-specified stepwise ablation of the proposed SE model:

1. paper reference control;
2. SE with the original last-state readout;
3. record-balanced training;
4. mean pooling across the causal UniLSTM sequence;
5. training-only contiguous ECG masking.

All candidates use the same five inner GroupKFold splits inside outer fold 0's
development partition. The excluded outer-test indices could not be read by
that runner.

The completed v3 evidence rejected record-balanced sampling and temporal mean
pooling. Its SE-last candidate improved mean AP by only 0.00575 and did not pass
the second-seed gate. See `docs/V3_EVIDENCE_DECISION.md` for the locked result.

## Historical focused validation-only quality sprint v4

Version 4 keeps the original last-state SE-CNN-HRV-UniLSTM and runs a complete
2x2 ablation of two training-only factors: contiguous ECG masking and BCE
versus focal loss under natural sampling. This directly tests the one useful
v3 signal without carrying forward the two harmful changes. All deployment
models remain at 159,503 parameters, and the paper control remains at 158,033.

All five candidates used the same five inner GroupKFold splits inside outer
fold 0's development partition. The excluded outer-test signals and HRV could
not be read by the runner. The current v5 procedure is in `COLAB_RUNBOOK.md`.

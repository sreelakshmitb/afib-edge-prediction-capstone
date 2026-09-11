# Completed AFDB experiment audit — 2026-09-11

All five outer folds completed for both candidates. No training or inference was restarted during this audit. All 72 existing run artifacts retained identical SHA-256 hashes and modification times during verification. No Python process was running at inspection.

Run artifacts: `C:/Users/Lenovo/afib-edge-data/runs/first-fold-20260911`. The run name is historical; it contains all five folds. Smoke runs under `smoke-20260911` are excluded from these results.

Training source commit: `7ccdad5a5ec3f5ea0f1f1f8043379eb022e6b051`. Manifest SHA-256: `19044e7b65a1db5b2e51f2a8a464fb4e69b68e304e1dcafd4515620a4312f0e3`.

## Fixed experiment

665 observations from 20 eligible patients: 573 negative and 92 positive. All 23 signal-bearing AFDB records were processed; three had no eligible observations. Single-lead 250 Hz ECG, 10-minute observations, 20 chronological 30-second segments, ECG `[B,20,1,7500]`, HRV `[B,20,6]`. HRV order: RMSSD, SDNN, LF/HF, Mean RR, pNN50, Sample Entropy. The target is AF onset within the next 20 minutes; observations overlapping AF are excluded.

Patient-wise GroupKFold(5); inner GroupShuffleSplit (25% of development patients, seed 2026). Each fold has 12 training, four validation and four test patients. Both candidates use identical splits and preprocessing. HRV imputation/scaling is fitted only on inner-training patients.

Maximum 20 epochs, patience 5, AdamW learning rate 5e-4, weight decay 1e-4, batch size 8, training-class-weighted BCE, gradient clipping at 1. Checkpoints selected by minimum unweighted inner-validation BCE. Fixed probability threshold 0.5. No architecture choice, threshold tuning or calibration used outer-test results.

The CNN-HRV-UniLSTM is a **PAPER REPRODUCTION / REFERENCE BASELINE**, attributed to Zhang et al. Unspecified details and departures from the paper remain **REPRODUCTION CHOICE** (see `PAPER_REPRODUCIBILITY_AUDIT.md`). SE-CNN-HRV-UniLSTM adds the **CAPSTONE EXTENSION**. The final deployed architecture remains undecided.

## Fold results

All metrics below use the saved outer-test predictions. AP means average precision. Confusion matrices are `[[TN, FP], [FN, TP]]`. Epochs are one-based; folds retain their saved zero-based numbering. Undefined precision is recorded as zero, matching the existing scorer.

### reference

Parameters: 187393.

| Fold | Selected / run epochs | N / positive | Accuracy | Precision | Recall | F1 | Specificity | AUROC | AP | Test BCE | Confusion matrix |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 4 / 9 | 133 / 9 | 0.932331 | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 0.512545 | 0.072001 | 0.271360 | [[124, 0], [9, 0]] |
| 1 | 20 / 20 | 129 / 16 | 0.736434 | 0.178571 | 0.312500 | 0.227273 | 0.796460 | 0.672013 | 0.196698 | 0.520574 | [[90, 23], [11, 5]] |
| 2 | 2 / 7 | 134 / 17 | 0.843284 | 0.333333 | 0.235294 | 0.275862 | 0.931624 | 0.661639 | 0.262663 | 0.476540 | [[109, 8], [13, 4]] |
| 3 | 7 / 12 | 134 / 28 | 0.798507 | 1.000000 | 0.035714 | 0.068966 | 1.000000 | 0.599057 | 0.304718 | 0.502016 | [[106, 0], [27, 1]] |
| 4 | 4 / 9 | 135 / 22 | 0.740741 | 0.240000 | 0.272727 | 0.255319 | 0.831858 | 0.530169 | 0.262302 | 0.572760 | [[94, 19], [16, 6]] |

Pooled out-of-fold confusion matrix: `[[523, 50], [76, 16]]`.

### se

Parameters: 198425.

| Fold | Selected / run epochs | N / positive | Accuracy | Precision | Recall | F1 | Specificity | AUROC | AP | Test BCE | Confusion matrix |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 4 / 9 | 133 / 9 | 0.932331 | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 0.529570 | 0.074785 | 0.252708 | [[124, 0], [9, 0]] |
| 1 | 3 / 8 | 129 / 16 | 0.821705 | 0.315789 | 0.375000 | 0.342857 | 0.884956 | 0.615044 | 0.273240 | 0.480451 | [[100, 13], [10, 6]] |
| 2 | 3 / 8 | 134 / 17 | 0.746269 | 0.257143 | 0.529412 | 0.346154 | 0.777778 | 0.654098 | 0.257309 | 0.574873 | [[91, 26], [8, 9]] |
| 3 | 12 / 17 | 134 / 28 | 0.798507 | 1.000000 | 0.035714 | 0.068966 | 1.000000 | 0.561658 | 0.277848 | 0.605723 | [[106, 0], [27, 1]] |
| 4 | 4 / 9 | 135 / 22 | 0.674074 | 0.225000 | 0.409091 | 0.290323 | 0.725664 | 0.533789 | 0.295281 | 0.602635 | [[82, 31], [13, 9]] |

Pooled out-of-fold confusion matrix: `[[503, 70], [67, 25]]`.

## Five-fold mean ± sample standard deviation

Unweighted mean over five folds; sample standard deviation uses ddof=1. These are descriptive fold summaries, not confidence intervals. Windows within patients are correlated.

| Metric | Reference | SE |
|---|---|---|
| accuracy | 0.810259 ± 0.081261 | 0.794577 ± 0.095632 |
| precision | 0.350381 ± 0.383003 | 0.359586 ± 0.377478 |
| recall | 0.171247 ± 0.143220 | 0.269843 ± 0.237411 |
| f1 | 0.165484 ± 0.123260 | 0.209660 ± 0.163272 |
| specificity | 0.911989 ± 0.094400 | 0.877679 ± 0.125565 |
| auroc | 0.595085 ± 0.073127 | 0.578832 ± 0.054145 |
| average_precision | 0.219677 ± 0.091150 | 0.235693 ± 0.090962 |
| test_loss | 0.468650 ± 0.115795 | 0.503278 ± 0.148996 |

## FP32 ONNX

Only fold 0 was exported for each candidate. These are experimental exports of inner-validation-selected checkpoints, not a final deployment selection. Existing parity reports confirm batches 1 and 2; this audit also passed ONNX graph checking without changing or re-exporting the files.

| Candidate | Parameters | Bytes | Maximum absolute parity error |
|---|---|---|---|
| reference | 187393 | 753725 | 5.96046448e-08 |
| se | 198425 | 802871 | 1.1920929e-07 |

Exports embed the training-fitted HRV transform. ECG filtering and observation normalization remain external. No Raspberry Pi latency or hardware measurements exist.

## Verification and limitations

- All ten required checkpoint/config/history/split/test-metric/test-prediction sets exist and parse successfully. Checkpoint weights load strictly into their corresponding models; parameter counts are below 250,000. Optimizer states contain completed updates.
- Saved classification metrics were independently recomputed from saved probabilities to absolute tolerance 1e-12. BCE reconstructed from saved sigmoid probabilities agrees with saved logit-based BCE within rtol 1e-5 / atol 1e-6. No outer-test inference was repeated.
- Checkpoint epoch and validation loss match the first minimum of each saved history. Every run reached epoch 20 or five non-improving epochs; all saved training losses, validation losses and gradient norms are finite.
- Regenerated splits exactly match saved indices and patients. Every observation appears in the outer test exactly once per candidate, and candidate split files are byte-identical. Checkpoint HRV medians, means and scales exactly match recomputation using only training observations.
- All 72 existing artifact hashes and modification times are unchanged across this resume audit. The trainer rejects existing fold directories using `exist_ok=False`. No pre-resume cryptographic snapshot exists to conclusively prove historical non-overwriting; current artifact consistency and this audit establish preservation during this resume. Normal replacement of best.pt within a single training run is intentional.
- Test suite: 17 passed. No raw ECG, generated arrays, checkpoints or ONNX files are included in this report commit.
- Predictive performance is weak and variable. At threshold 0.5 the reference detects 16/92 positive observations, and SE detects 25/92. Neither result establishes deployment readiness or superiority. Precision of 1.0 in fold 3 reflects only one predicted positive.
- This is a small, single-dataset experiment with one fixed inner holdout and seed per fold. It is not an exact reproduction of paper results. LF/HF uses trailing five-minute context, early segments have missing values, and detected RR intervals have not been adjudicated as normal-to-normal beats. Conservative rhythm eligibility restricts the evaluated population; see the preprocessing protocol.

## Artifact fingerprints

SHA-256 fingerprints captured at this resume audit (not a historical pre-training baseline).

| Artifact | SHA-256 |
|---|---|
| reference-fold0/best.export.json | `2bdfdad0ee3c19d11da56d6c162e34964de078f59ff5d323ea591129d7cac3ac` |
| reference-fold0/best.onnx | `2a1543afbf1cc4c1d102dae9cbffc0e94c5766b7f11bf7dae3aa48885691e15b` |
| reference-fold0/best.pt | `365577cb6a5bbe72057b861b61767b0945dcdb8eb42c8d19197a17369befa182` |
| reference-fold0/test_metrics.json | `6a75cc264f7435a84149ded5c17a47528477847574ae163c23a2eef6dea44fcc` |
| reference-fold0/test_predictions.json | `5305b6745529a0b74761493eabdbf4d11ea37159387c62874bd511570153083a` |
| reference-fold1/best.pt | `416f6cc2c28175601f412cb3c14a0f5e8505419f9509b0d13f049ce1f205d136` |
| reference-fold1/test_metrics.json | `53035f15272fe89a70fc45d78f665a2bdca4f477a7ce360615e8f5b9c56bedd9` |
| reference-fold1/test_predictions.json | `8139f47ad6c34d4b391d533aed5b3c5d27dc8f638b1980b2a2fa9a7b10f4ee18` |
| reference-fold2/best.pt | `93295b0c48fa1de045d17d2bfb897a6716c10034b773feabb47e3a04c25f633e` |
| reference-fold2/test_metrics.json | `1689155dc528eec28ad9519d421756ad698f2785757b5480881278a6a47356a4` |
| reference-fold2/test_predictions.json | `9194d819c1c4e47c7a6975aa8a02fecbe53b3f53421525538cef72f322ed47ab` |
| reference-fold3/best.pt | `12a6710b9d14b2ee401bbca6382e3643fb4d19f6b4df6e4f0f61912648a0cb65` |
| reference-fold3/test_metrics.json | `ac4a2fa4d4e637b81ec5280b5fe95292426ac220e10b65e505bdc2466f7ff997` |
| reference-fold3/test_predictions.json | `0a3a9bf42074e636fb234f7e1a685b34345c6b25eea57a98169008aa404c9b30` |
| reference-fold4/best.pt | `563728f91c84f791cd482e6be7b13d8afd0adb51ed77a0b19b272a07ffee62c4` |
| reference-fold4/test_metrics.json | `dee08b9b591bdf71d58700903684c5d6526ce9491a3b9de6da44bf73aa1c362d` |
| reference-fold4/test_predictions.json | `553f5e14e4168ef517e18591def08678d160147369989ac5ca19b9abbb306186` |
| se-fold0/best.export.json | `e88ea9e9f59d74a3dcf9829dbaf0acdb60e4493cf748b712496554570d44424a` |
| se-fold0/best.onnx | `c242bb569b10d2ba663760b91e42445c92315f808f408a36f860ddc598d06210` |
| se-fold0/best.pt | `ff2617b1f42581af334f5830d6ac601e229c86ecd164ed97a67fa6d3e72dbb1c` |
| se-fold0/test_metrics.json | `5bae4b679c9d60033dde409af6103d3ef0c890ac2a6c95ccdee634b1806d9df9` |
| se-fold0/test_predictions.json | `3dad802503818ada90f2ddba7dde910a3b5dbca7608f51576a5bc585579e78aa` |
| se-fold1/best.pt | `4d9d88d47c8a399fcb8086ccb8335047a8f6022524d79658661bec9c678e479a` |
| se-fold1/test_metrics.json | `0f9b1ff938cca1f62de5cf5d8cc44453dcda8581e1d765ae6e55dff10bd4a11f` |
| se-fold1/test_predictions.json | `7cb013e018c6b6285de9cea519e9b1382ff47b02263547767ca48cdca2c4f01b` |
| se-fold2/best.pt | `4f357fe21c5d3591ff7b51c072175ae50958647560dac56cfaf12004af36135a` |
| se-fold2/test_metrics.json | `dc8bfe0d49f903493eb6548c190ee75ed5222d595ed778b2e9368f2073c2fb4a` |
| se-fold2/test_predictions.json | `9f84544d77189b8accb2dbe08d61b763ca6f7215ae4176869f04e649701c4fad` |
| se-fold3/best.pt | `186d248a5c9156cc8bc75926b13ada3dd3bf115f87a14716a71fc0962f7b1ac0` |
| se-fold3/test_metrics.json | `5859761e655367586939296903acc6a5e8b7889255c7c1620d541fa872b8bbe8` |
| se-fold3/test_predictions.json | `f2e49964ed3a0670cb19c229e08a33796d6217ab9b09865099d58756418bcd58` |
| se-fold4/best.pt | `bfa714a7a6c5c8704c0cb1921567635748e9f2c1d61de9280c4236b06b932a68` |
| se-fold4/test_metrics.json | `7864dbb6f59926ababab558e126c941604eb43680c1d6af41b37b18dc38b4a57` |
| se-fold4/test_predictions.json | `be589cc137ace93f33107ea85c708d7745c227be0d81f8baae9db87214bf5038` |

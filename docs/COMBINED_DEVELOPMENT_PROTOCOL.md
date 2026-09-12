# Combined-cohort development, distinct from PILOT_V1

The objective is lightweight AF-onset prediction on unseen patients, not exact
paper replication. Zhang et al. supplies a PAPER-BASED reference architecture;
our preprocessing, evaluation, adaptations and deployment work are engineering
contributions. The SE modification is a capstone extension. No final architecture
has been chosen.

## Preservation

PILOT_V1 is the completed AFDB-only experiment at source commit 7ccdad5 and
report commit 8c36169. Its external root manifest, processed arrays, ten fold
checkpoints and predictions remain untouched. Its historical run directory is
`C:/Users/Lenovo/afib-edge-data/runs/first-fold-20260911`.
The audit report contains existing checkpoint fingerprints.

New LTAFDB files use `C:/Users/Lenovo/afib-edge-data/ltafdb-v1`.
The future combined cohort must have its own manifest and run directories.
No existing PILOT_V1 evaluation is to be repeated or relabeled as a fresh test.

## LTAFDB verification gate

Source: https://physionet.org/content/ltafdb/1.0.0/ (version 1.0.0).
Use reviewed `.atr` rhythm annotations, not `.qrs` AF-termination markers.
Native sampling is 128 Hz; resample channel 0 to 250 Hz using observation ECG
only. Retain PILOT_V1's fixed observation grid and HRV rules initially. The new
onset-v2 label policy requires fully known follow-up but allows other known
future rhythms: they must not erase a subsequent AF onset. Record 00's reviewed
annotations exposed that the old follow-up exclusion removed every positive
candidate because VT co-occurred in those horizons. This is a target-definition
correction made before any combined-cohort model evaluation, not a test-driven
label adjustment. Apply it consistently to both datasets in new manifests.
Collapse repeated identical rhythm markers before onset
extraction. Check real-record onset boundaries, shapes, missingness, detected
peaks versus reviewed beats, and all source SHA-256 digests before bulk work.

AFDB `(N` means all rhythms outside its separately annotated AF/AFL/J classes;
it does not establish clean sinus/NN beats. Do not describe these windows as
independently confirmed pure sinus rhythm. Observations must be non-AF and meet
the documented annotation eligibility policy. RR-based HRV remains potentially
affected by ectopic beats and detector errors.

## Identity and scope

Use `afdb:<record>` and `ltafdb:<record>` subject groups. The public record
convention is one recording per subject; files do not provide a cross-dataset
person linkage. Namespacing prevents ID collisions but cannot establish that
two anonymized recordings belong to different people. Check source provenance,
duplicate source hashes and signal fingerprints; disclose the remaining linkage
limitation. Never claim an absolute biological identity guarantee from filenames.

AFTDB contains excerpts derived from LTAFDB and is therefore ineligible as an
independent final-validation dataset. No external final-validation data is to be
used during development.

## Selection boundaries

Preserve outer patient-wise GroupKFold(5). Establish all outer partitions before
model fitting. All preprocessing statistics use inner-training patients only.
Group-aware inner validation selects checkpoints, thresholds and candidates.
Selection must be repeated inside each outer fold: a global winner tuned on one
fold's development patients would contaminate other outer folds containing those
patients. Freeze the small candidate-selection algorithm before outer testing.

Start with the existing reference and SE candidates, both below 250,000
parameters. Diagnose labels, onset spacing, eligibility, class/patient balance,
peak quality, HRV missingness, preprocessing and optimization before adding any
third architecture. At most one additional justified temporal candidate is allowed.
Use a small prespecified training comparison; never adapt it to outer results.

Threshold selection must use only inner-validation predictions. Report recall,
F1, average precision (the chosen AUPRC estimator), AUROC and specificity.
Record threshold policy and freeze the numeric threshold before each outer test.
Meaningful validation performance cannot be promised in advance. Do not run or
claim a final evaluation merely because training completed. PILOT_V1 results
informed this follow-up, so combined-cohort results must disclose that history;
truly independent external validation remains necessary after final freezing.

## First targeted comparison (declared before combined training)

Three configurations only: reference + full training negative/positive BCE
weight; SE + the same weight; SE + square root of that weight. No weighted
sampling or focal loss in this first comparison, so imbalance correction is
not unintentionally doubled. AdamW lr 5e-4, weight decay 1e-4, batch size 8,
maximum 20 epochs, patience 5, gradient norm clip 1. These optimizer settings
are PAPER-BASED starting points; the targeted comparison is capstone development.

Use the first split of StratifiedGroupKFold(4, shuffle=True, seed=2026+outer_fold)
within each outer development partition. Stop if either inner partition lacks
a class. This preserves group separation while improving class coverage.
Checkpoint selection maximizes inner-validation average precision, with lower
unweighted BCE as a tie break. Threshold selection maximizes validation recall
subject to specificity >= 0.70, breaking ties by F1, specificity, then threshold.
When any threshold also meets F1 >= 0.25, restrict selection to those thresholds
before maximizing recall. Otherwise preserve the best specificity-constrained
failed operating point for diagnosis. This avoids discarding a candidate that
has a useful validation operating point in favor of one extra true positive
accompanied by many false alarms. This edge case is covered by a synthetic test
and was resolved before any combined-cohort fitting or validation prediction.

Before outer inference, each inner-selected candidate must have validation
recall >= 0.50, F1 >= 0.25, specificity >= 0.70 and average precision above that
validation set's prevalence. This is an engineering screening criterion, not a
clinical performance standard or guarantee. Among passing candidates choose by
recall, F1, AP, AUROC, specificity, in that order. If none passes, preserve the
best failed result for diagnosis and do not open outer tests. All five inner
folds must finish and pass before the separate evaluate phase is allowed.

Each candidate saves resumable epoch state and a completion marker; completed
candidates and outer results are preserved. Manifest, model-source, training-
source and plan fingerprints must match when resuming. No final-validation
dataset is accessed by this development command.

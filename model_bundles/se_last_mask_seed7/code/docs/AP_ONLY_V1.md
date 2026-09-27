# AP-Only V1: Natural BCE Plus Smooth-AP

## Motivation

The preceding five-fold experiment produced two separable observations:

- equal record/patient weighting reduced mean validation AP by `0.016513`
  versus the masked-BCE baseline; and
- adding Smooth-AP to that weighted objective recovered `0.010380` AP versus
  weighting alone.

The combined model still lost to the baseline because it retained the harmful
weighting. This exploratory V9 sprint tests the unconfounded candidate that was
not present in V8: natural masked BCE plus the same fixed Smooth-AP term.

Smooth-AP replaces hard pairwise ranking indicators with a sigmoid relaxation,
allowing gradient-based optimization of an AP approximation. The implementation
here uses the already-tested temperature `0.1` and coefficient `0.2`.

## Locked comparison

| Candidate | Objective | Training weights |
|---|---|---|
| `se_last_mask` | BCE | Natural, one per window |
| `se_last_mask_ap` | BCE + 0.2 × Smooth-AP | Natural, one per window |

Both candidates use the same:

- 10-minute observation and 20-minute prediction target;
- 159,503-parameter masked SE-CNN-HRV-UniLSTM;
- five development-only GroupKFold splits;
- HRV preprocessing fitted on each inner-training split;
- ECG masking augmentation;
- optimizer, learning rate, dropout, seed, batch size, patience, and epoch cap;
- unweighted validation AP at natural prevalence.

No outer-test signal, HRV value, label, threshold, or prediction is accessed.

## Objective behavior

Smooth-AP is evaluated inside each shuffled minibatch. If a batch lacks either
class, the AP term is differentiable zero and the candidate uses ordinary BCE
for that batch. Validation and checkpoint selection use exact unweighted AP,
not the training surrogate.

## Decision rule

`se_last_mask_ap` is promoted to a seed-19 confirmation only if, across all five
folds, it achieves:

1. mean paired AP improvement greater than `0.01`;
2. at least three fold wins; and
3. non-negative mean paired within-record AUROC change.

The test is explicitly exploratory because it was motivated by the completed
V8 factorial result. If it fails, the objective-only path is exhausted. The
next credible source of improvement is additional prediction-aligned onset
data rather than more tuning of these same development folds.

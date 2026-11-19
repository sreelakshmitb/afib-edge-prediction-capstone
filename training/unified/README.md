# Unified AF Forecast Training v1

This package locks the post-IRIDIA training protocol before any new model result
is observed. It combines 4,862 legacy AFDB/LTAFDB windows with the locked 3,430
IRIDIA cache windows through lazy memory maps; ECG arrays are never duplicated.

The protocol uses five patient-disjoint outer folds and a separately frozen
patient-disjoint inner validation split inside every outer fold. The existing
legacy locked-final run is retained as historical baseline evidence, and CPSC
is not included in training or model selection.

Training batches contain exactly eight positives and 24 negatives. Each label
is split equally between legacy and IRIDIA sources. Positive draws are balanced
by patient, event and four-minute lead-time bin; IRIDIA negatives are stratified
near/medium/far. HRV imputation and scaling are fitted only on the inner-training
patients. Validation and test loaders will use each cached window once without
balanced resampling.

The first run intentionally stops after protocol locking and one GPU
forward/backward batch. Version 1.1 added a validation-only fold-0 pilot. Version
1.2 adds two predeclared, controlled repairs after the pilot exposed compressed
scores and a strong source shift: GroupNorm alone, followed by GroupNorm plus a
near-to-far lead-time curriculum. Both use the same frozen fold, seed, optimizer,
loss and balanced batch contract. The comparison script keeps outer evaluation
locked unless a candidate passes every predeclared ranking, operating-point,
domain and patient-coverage gate. Version 1.2.1 hardens the sampler so a window
can never be selected twice within one curriculum batch, including after the
patient-balancing counters roll over. Version 1.3 adds independently trained
ECG-only and HRV-only curriculum ablations. These diagnostic runs distinguish
true single-branch learnability from the harsher post-hoc masking result and
cannot unlock outer evaluation. Version 1.4 adds a single targeted full-fusion
repair: label-conditional mean alignment of legacy and IRIDIA temporal
representations. Because training batches are already exactly source-balanced
inside each label, the alignment penalty targets source routing without erasing
the AF class boundary. Version 1.4.1 normalizes NumPy gate booleans to native
Python values before comparison evidence is serialized; trained checkpoints are
unchanged.

Version 1.5 replaces the single 20-minute binary head with a discrete-time
survival head over five clinically interpretable intervals: 0--4, 4--8,
8--12, 12--16 and 16--20 minutes. Exact AF onset timing is joined back onto the
immutable cache-addressable manifest, with hash, event, lead-bin and censoring
checks. The model emits five conditional hazards and converts them to
cumulative risk, which is monotonic by construction. Training retains the exact
8-positive/24-negative batches, source balance, patient/event/lead-bin balance,
and negative proximity strata. Its loss additionally normalizes each window's
observed intervals and gives positive and censored-negative window means equal
weight, preventing the five negative survival terms from dominating. This is a
single structural validation experiment on the frozen fold; it does not access
or unlock the outer test.

Version 1.6 corrects a legacy-manifest omission discovered after the structural
experiment: AFDB/LTAFDB negatives had no next-onset or previous-AF timing, so
all were treated as unstratified and post-AF recovery windows remained eligible.
The correction reads rhythm annotations only, preserves every cache address,
verifies every positive onset, excludes observations beginning within 30
minutes after AF ends, and assigns the same near/medium/far negative strata used
for IRIDIA. The controlled repair keeps the architecture and optimizer fixed,
stratifies both sources, and compares old and new models on the identical
corrected validation rows. No ECG cache is rebuilt.

# Initial AFDB protocol

Source: https://physionet.org/content/afdb/1.0.0/ (version 1.0.0).
The two annotation-only records 00735 and 03665 cannot supply ECG and are excluded.
Use channel 0 consistently, record its original lead name, and group by AFDB record
identifier (the dataset's subject identifier convention). Do not pool another
dataset without an explicit subject mapping.

Use non-overlapping 600-second observations on a fixed record-relative grid.
Keep only entirely annotated normal rhythm `(N` observations. Target is any
annotated AFIB onset in [observation end, end + 1200 seconds], with complete
follow-up. Unknown rhythm, atrial flutter and junctional rhythm in observation
or follow-up exclude the example. No minimum AF episode duration is imposed.
At the exact observation endpoint an onset is a positive label, not observation
overlap; sample intervals are half open. This is recurrence/onset prediction
within an AF-enriched cohort, not population incident-AF risk prediction.

Read only observation ECG for feature computation. Resample only when native
rate differs from 250 Hz; apply a fourth-order Butterworth 0.5–40 Hz causal SOS
filter. Detect R peaks using WFDB XQRS within the available observation. Reject
nonfinite signals, flat runs of at least one second, fewer than 200 detected
peaks, and fewer than ten segments with valid time-domain HRV. XQRS operates
offline within the ten-minute observation; no horizon ECG is supplied.
Normalize ECG by its observation mean and standard deviation, without population
statistics. This normalization is available at the prediction time.

Time-domain HRV uses RR intervals wholly within each 30-second segment.
RR values outside 0.3–2.0 seconds make that segment's time-domain estimates
missing; do not bridge invalid intervals. At least ten RR intervals are needed.
RMSSD, sample SDNN and Mean RR are in milliseconds; pNN50 is a percentage.
Sample Entropy uses m=2, tolerance 0.2 times sample standard deviation,
Chebyshev distance, equal template counts and no self matches; undefined results
are missing, not zero.

LF/HF uses the trailing five minutes wholly inside the observation: interpolate
RR at 4 Hz, Welch PSD with up to 1024 samples, integrate LF 0.04–0.15 Hz and HF
0.15–0.4 Hz. Require >=290 seconds RR coverage and no invalid RR. Thus early
segments intentionally have missing LF/HF. These choices avoid claiming that
30-second LF/HF is a reliable conventional short-term estimate. XQRS peaks are
not beat-type adjudicated; the resulting features are RR-based HRV estimates,
not guaranteed clean NN estimates. Short-segment entropy is also exploratory.

Feature order: RMSSD, SDNN, LF/HF, Mean RR, pNN50, Sample Entropy.
Missing values must be imputed and scaled using inner-training patients only.
No quality, label or sampling rules may be tuned against outer-test performance.

Generated raw files, arrays, manifests, checkpoints and run logs live under an
explicit root outside the Git checkout. Each record report preserves source
SHA256, sampling rate, lead, rhythm intervals, exclusions and class counts.
Acquisition defaults to resumable four-megabyte ranges from PhysioNet's official
public S3 mirror because an observed WFDB signal transfer stalled without a
timeout. WFDB still reads all ECG and rhythm annotations. Every source file is
verified against the published AFDB SHA256SUMS on archive.physionet.org before
processing. Connections are reused within each record. The optional
`--source wfdb` uses WFDB's own downloader. At most two records are processed
concurrently (`--workers 2`); manifest ordering remains deterministic.

Example (PowerShell, replace the root with your external experiment folder):

```powershell
.venv/Scripts/python.exe -m afib.preprocess --root C:/afib-data --records 04015
.venv/Scripts/python.exe -m pytest -q
```

The supplied primary paper has been inspected. See PAPER_REPRODUCIBILITY_AUDIT.md
for supported architecture details and explicitly documented reproduction gaps.

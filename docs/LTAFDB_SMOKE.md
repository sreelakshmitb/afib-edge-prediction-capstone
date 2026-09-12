# Real LTAFDB record 00 verification

Source: [PhysioNet LTAFDB 1.0.0](https://physionet.org/content/ltafdb/1.0.0/).
Use its reviewed `.atr` beat/rhythm annotations, acknowledging MEDICALgorithmics
and the original database publication. `.qrs` termination markers are not onset
labels. This verification is capstone data engineering, not paper replication.

The real header specifies two ECG channels, 128 Hz and 9,661,440 samples.
Only channel 0 is supplied to preprocessing. Its 600-second observations are
resampled to 150,000 samples, producing ECG `[20,1,7500]` and HRV `[20,6]` per
observation. The entire prediction horizon is excluded from signal processing.

The onset-v2 output contains 100 observations: 98 negative and two positive.
Six additional eligible observations failed the fixed RR-quality rule; 20 grid
positions were excluded by rhythm or follow-up eligibility. The old pilot
follow-up rule excluded both positive observations because other known rhythms
also occurred in their prediction horizons. PILOT_V1 remains unchanged.

## Manual boundary check against the annotation samples

The first relevant AF transition is at sample 4,613,049, or 36,039.4453125 seconds.

| Observation interval, seconds | Prediction time | AF lead time, seconds | Horizon end | Label |
|---|---:|---:|---:|---:|
| [34800,35400) | 35400 | 639.4453125 | 36600 | 1 |
| [35400,36000) | 36000 | 39.4453125 | 37200 | 1 |

Both observations precede AF and both horizons contain the onset. These are two
positive windows associated with the same first onset, not two independent
patients or necessarily two independent AF events. Repeated identical rhythm
markers are collapsed before identifying transitions. Unknown annotation state
at the beginning of the record is censored, not assumed normal.

## Programmatic evidence

`afib.verify_ltaf` checked all retained labels against an independent transition
scan, observation length, group identifier, shapes, finite ECG and minimum HRV
availability. Recomputed observation-only filtered/resampled ECG matched saved
tensors to absolute tolerance 1e-6.

One negative and one positive observation were compared with reviewed beat
annotations using one-to-one matching within 100 ms. Reviewed annotations were
used only for this diagnostic, never as model HRV inputs.

| Observation start sample | Class | Detected beats | Reviewed beats | Matched | Sensitivity | Precision |
|---|---:|---:|---:|---:|---:|---:|
| 153600 | 0 | 759 | 730 | 716 | 0.980822 | 0.943347 |
| 4454400 | 1 | 768 | 761 | 758 | 0.996058 | 0.986979 |

Missing fractions in fixed HRV order: RMSSD 0.1085, SDNN 0.1085, LF/HF 0.6800,
Mean RR 0.1085, pNN50 0.1085, Sample Entropy 0.3595. The first nine LF/HF segments
are intentionally missing because five minutes of observation history are not
yet available. These results do not prove detector quality across the database.

Header, annotations and signal matched published SHA-256 checksums. Record
identity and the `ltafdb:00` namespace were verified. No biological person ID
linking AFDB and LTAFDB is public; that linkage limitation remains explicit.

The immutable smoke outputs are under
`C:/Users/Lenovo/afib-edge-data/ltafdb-v1/processed-onset-v2/00`.
Manifest SHA-256:
`4cb53ba496b95743cefaaa8522c33c5303bc1bf1b35c5d9d7d063d3d4767bf3c`.
`verification.json` records the exact checks and source fingerprints.

The scientific diagnostic image (rhythm timeline, ECG and HRV missingness) is
outside Git at
`C:/Users/Lenovo/.codex/visualizations/2026/09/11/01a08f89-a0ff-7272-ac4d-4f8032c5bd8c/ltaf00-diagnostic.png`.

## Annotation-first acquisition policy

The signal is downloaded whenever at least one observation is eligible under
the fixed onset-v2 policy, even if every label is negative. Zero-window records
retain empty manifests, reports and exclusions. The combined screening audit
also retains AFDB's two annotation-only records, 00735 and 03665. No patient is
removed to improve class balance. Unit tests cover both negative-only retention
and sustained-AF signal skipping. Existing partial downloads resume by byte range.

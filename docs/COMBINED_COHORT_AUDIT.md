# Completed combined onset-v2 cohort

The existing LTAFDB acquisition resumed without reprocessing completed records
and exited successfully. All 84 records have completion reports; no partial
downloads remain. The real-record 00 scientific verification was preserved.
The combined builder exited successfully after source SHA-256 checks, duplicate
source detection, annotation-based label checks, tensor validation, ECG
normalization checks, HRV accounting, and patient-split checks.

## Cohort counts

| Database | Screened groups | Groups with windows | Windows | Negative | Positive | Prevalence |
|---|---:|---:|---:|---:|---:|---:|
| AFDB | 25 | 20 | 674 | 577 | 97 | 14.3917% |
| LTAFDB | 84 | 47 | 2,867 | 2,686 | 181 | 6.3132% |
| Combined | 109 | 67 | 3,541 | 3,263 | 278 | 7.8509% |

All 42 zero-window groups remain in the audit, including the two AFDB records
without ECG. All 14 negative-only contributing groups remain included.
Quality rejections and rhythm/follow-up exclusions are recorded per source
group in the external audit. No patients were dropped to improve class balance.
AFDB reused all 665 PILOT_V1 observations with unchanged labels and added only
nine newly eligible observations (four negative, five positive).

## HRV missingness

Percent of segment-level feature values missing before training-only imputation:

| Feature | AFDB | LTAFDB | Combined |
|---|---:|---:|---:|
| RMSSD | 3.8724% | 2.0875% | 2.4273% |
| SDNN | 3.8724% | 2.0875% | 2.4273% |
| LF/HF | 56.8769% | 50.6906% | 51.8681% |
| Mean RR | 3.8724% | 2.0875% | 2.4273% |
| pNN50 | 3.8724% | 2.0875% | 2.4273% |
| Sample Entropy | 28.4570% | 27.2951% | 27.5162% |

LF/HF uses trailing five-minute RR context within the observation; the first
nine segments intentionally have no LF/HF estimate. Missingness also reflects
quality and estimator requirements. These are RR-derived features, not a claim
of independently adjudicated clean NN intervals.

## Patient splits

Outer GroupKFold(5), with inner StratifiedGroupKFold(4), passed disjoint-group
checks and exactly-once outer coverage. Counts below are windows (positives).

| Fold | Train | Inner validation | Outer test | Patient groups train/validation/test |
|---|---:|---:|---:|---|
| 0 | 2,143 (176) | 690 (57) | 708 (45) | 39 / 15 / 13 |
| 1 | 2,127 (153) | 706 (50) | 708 (75) | 40 / 13 / 14 |
| 2 | 2,146 (192) | 687 (58) | 708 (28) | 42 / 12 / 13 |
| 3 | 2,085 (153) | 748 (57) | 708 (68) | 39 / 15 / 13 |
| 4 | 2,100 (153) | 732 (63) | 709 (62) | 40 / 13 / 14 |

No duplicate source recording hashes were found. Groups follow namespaced
record identity; public files do not provide cross-database person linkage.
Therefore 67 groups is not proof of 67 distinct biological identities across
both databases. This limitation must accompany generalization claims.

## Artifacts and preservation

External cohort root: `C:/Users/Lenovo/afib-edge-data/combined-onset-v2`.
`manifest.json`, `cohort-audit.json`, and `splits.json` contain the complete
machine-readable outputs. Arrays, raw signals and training outputs remain
outside Git.

- Combined manifest SHA-256: `14d1e61e9eb3893a3eed922488f5576aac092b6892b67e285a968d9664f6f823`
- Preserved PILOT_V1 manifest SHA-256: `19044e7b65a1db5b2e51f2a8a464fb4e69b68e304e1dcafd4515620a4312f0e3`
- Core preprocessing SHA-256: `8c491b652b2ec3fd118d67832e663f0a5fb38a5fb5d9974cbc8f923a6a7483be`

All 72 saved pilot artifacts matched their prior hashes and modification times.
All 30 tests passed before acquisition resumed. No completed pilot fold was
restarted. Neither final external-validation data nor combined outer-test
predictions were used for development.

The prespecified three-configuration `targeted-v1` inner-only development run
was launched immediately after the cohort audit. Its training outputs must be
inspected separately; this report makes no predictive-performance claim.

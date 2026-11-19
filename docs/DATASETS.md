# Datasets and roles

The four forecasting development datasets were AFDB, LTAFDB, IRIDIA-AF and SHDB-AF. Their role varies by experiment; all four must not be attributed to every checkpoint.

| Dataset | Role | Project audit/cache count |
| --- | --- | --- |
| [MIT-BIH Atrial Fibrillation Database](https://physionet.org/content/afdb/) | Legacy forecasting | 25 nominal headers, 23 usable signal/annotation pairs; 00735/03665 lack .dat |
| [Long Term AF Database](https://physionet.org/content/ltafdb/1.0.0/) | Legacy forecasting, 128 to 250 Hz | 84 usable; combined retained legacy cache: 67 records, 4,862 windows, 368 positive |
| IRIDIA-AF | Unified forecasting alongside legacy | Reported 167 metadata records / 152 patients, 166 cached; lead 0, 200 to 250 Hz; historical eligible set 3,430 windows / 686 positive |
| SHDB-AF | Later three-source BCE development training | Reported extraction 62/65 planned records, 5,876 windows / 1,182 positive; final repaired manifest needed |

These are project cache/audit counts, not full published dataset sizes or necessarily frozen-checkpoint training counts. The R1 test identifies legacy and IRIDIA. SHDB belongs to later development; recover exact manifests before attributing it to the frozen checkpoint.

Icentia11k and AFPDB supported exploratory morphology/sequence pretraining and transfer that did not replace the final model. Icentia's small morphology cache had 1,701 segments from 50 patients (984 normal, 717 AF/flutter); AFPDB used 25 pair groups.

CPSC2021 was screened for eligibility. The strict gate found no qualifying example with ten preceding normal minutes and sustained AF/AFL; no CPSC waveform was scored for this project's frozen task. RR-only Figshare work was exploratory.

Patient separation is mandatory: GroupKFold(5) outer partitions and group-aware inner holdouts; training-only preprocessing statistics. Adjacent windows must not cross patient partitions. Exact populations require locked manifests and split files, absent from these final archives.

No complete raw dataset is redistributed. Follow original providers' access and attribution terms. Three supplied deployment fixtures are preprocessed examples, not a training reproduction or general benchmark.

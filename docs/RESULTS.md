# Results and interpretation

Values come from supplied result files. This update does not retrain or tune the model.

| Experiment | Population | AP | AUROC | Meaning |
| --- | --- | ---: | ---: | --- |
| R1 frozen sealed outer fold 0 | 1,809 windows / 236 positive | 0.221747 | 0.664241 | Primary deployed-checkpoint test |
| R3 frozen validation | 1,664 windows / 185 positive | 0.223521 | 0.659723 | Used for operating-point selection |
| R2 later BCE fivefold development | 4,635 pooled windows / 633 positive | 0.250891 mean fold AP | 0.673000 mean fold AUROC | Different checkpoints/protocol |

## R1: held-out test

Source: [original summary](../results/frozen_forecast_sealed_test_v1/summary.json) and [predictions](../results/frozen_forecast_sealed_test_v1/test_predictions.csv). Threshold 0.3310546875, frozen from validation.

| Measure | Overall | IRIDIA | Legacy AFDB/LTAFDB |
| --- | ---: | ---: | ---: |
| Windows | 1,809 | 606 | 1,203 |
| Positive / negative | 236 / 1,573 | 127 / 479 | 109 / 1,094 |
| AP | 0.221747 | 0.272556 | 0.177202 |
| AUROC | 0.664241 | 0.606250 | 0.649187 |
| TP / FN / FP / TN | 55 / 181 / 142 / 1,431 | 44 / 83 / 100 / 379 | 11 / 98 / 42 / 1,052 |
| Recall | 23.31% | 34.65% | 10.09% |
| Specificity | 90.97% | 79.12% | 96.16% |
| Events detected / eligible | 38 / 107 | 30 / 52 | 8 / 55 |
| 16–20 min events detected / eligible | 9 / 48 | 5 / 26 | 4 / 22 |

Derived precision 27.92%, F1 25.40%, accuracy 82.14%. Class imbalance makes accuracy alone inadequate. Event counts are not false alarms per hour. Low sensitivity and source differences limit use.

## R2: later fivefold development

[Aggregate](../results/far_lead_group_cv_v1/summary.json) and five `fold*/bce_control/result.json` files preserve original evidence. BCE pooled counts: TP 183, FN 450, FP 585, TN 3,417; events 112/279; far events 41/150. Source/fold cutoffs can differ and were learned from inner holdouts. Far weighting reduced mean AP to 0.240258, detected events to 90/279 and far events to 34/150, and did not replace the frozen model. Only BCE per-fold files were supplied; the aggregate contains the far-weighted comparison.

## R3: frozen validation

[Summary](../artifacts/frozen_forecast_v1/forecast_summary.json), [predictions](../artifacts/frozen_forecast_v1/forecast_validation_predictions.csv): TP 55, FN 130, FP 144, TN 1,335 at 0.3310546875; recall 29.73%, specificity 90.26%, F1 28.65%. These are validation, not independent test results.

## Recorded Pi replay

Prior evidence audit recorded checksum-verified 600,000-byte transfers and Pi inference: POS102 **0.35684036**, NEG024 **0.05910458**. These curated demonstrations show the replay path, not generalization accuracy. Short live waveform evidence and reported operation do not replace a measured ten-minute live forecast/reliability study.

Earlier September metrics remain historical and must not be combined with R1 or presented as deployed-checkpoint fivefold results.

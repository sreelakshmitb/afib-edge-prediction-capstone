# Current team project bundle

Imported from `afib_best_prediction_model_se_last_mask_seed7.zip`, supplied by
the team. Sreelakshmi, Jagadjith, and Aravind jointly completed the capstone work.

Archive SHA-256:
`681015f133b1e6a1d619852be2feebe574086ea3bbc91b92fe767e39ceea66e0`.

All 48 archive members are preserved byte-for-byte under
`model_bundles/se_last_mask_seed7/`. The bundle manifest lists checksums for 45
members; the model card, loading notes, and manifest itself are the other three.
All 45 listed checksums were verified. Git attributes disable newline conversion
inside the bundle. The five checkpoints are approximately 665 KB each and are
intentional exceptions to the general checkpoint ignore rule. No raw ECG arrays
or signals are included.

The supplied model card reports 159,503 parameters, mean inner-validation AP
0.182597, pooled out-of-fold AP 0.153119, and pooled AUROC 0.599308. These are
supplied experimental results, not independently rerun training results or final
outer-test performance. The five inner folds belong to the development partition
of outer fold 0; they must not be described as five completed outer folds.

The supplied README references several Colab runbooks and tests that are absent
from the archive. This upload does not invent or reconstruct those missing files.
Its historical preprocessing differs from the root `afib/` implementation; use
the bundled source and saved HRV transform with these checkpoints.

The model card's within-record AUROC is 0.575418, whereas the bundled
`evidence/oof_summary.json` reports 0.574009 for `se_last_mask`. Both original
artifacts are preserved; this discrepancy has not been resolved by rerunning
experiments. Neither value should be silently substituted for the other.

Earlier local experiments, PILOT_V1, and uncommitted diagnostic work are
preserved. No external dataset, preprocessing, training, or outer-test inference
was started as part of publishing this bundle.

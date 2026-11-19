# Repository consolidation — 20 November 2026

## Imported sources

| Supplied artifact | Destination |
| --- | --- |
| AFib_thesis_model_results_and_checkpoint.zip | artifacts/frozen_forecast_v1 and results |
| afib_pi_deployment_frozen_v1.zip | deployment/pi/model |
| afib-unified-training-v1.7-corrected-eligibility.zip | training/unified and training/archives |
| afib-finalization-fast-v1.zip | training/finalization |
| AFibForecast_Android_Live_Resume_v1.4.8.zip | android |
| live_ecg_pi_waveform.py, pi_app_bridge.py, demo_replay(1).py, wifi_replay_receiver(1).py | deployment/pi, canonical script names |
| afib_live_wifi.ino | firmware/afib_live_wifi |
| AFIB_Carrier_RevB_Prototype_Fab_Candidate (2).zip, nested ZIP | hardware/archives and hardware/pcb |
| AFib_CPU_Audit_and_Export.ipynb, AFib_Prediction_Audit_2026-10-15.ipynb | notebooks, outputs cleared |

The 19 November master handover was used to interpret experiment roles and recorded tests. Original numeric JSON/CSV files and frozen model bytes remain intact. File-level identities are in [RELEASE_MANIFEST.json](../artifacts/RELEASE_MANIFEST.json). The manifest covers frozen evidence, not every editable source file.

## Cleanup decisions

Root README and project state now identify the actual frozen model. The obsolete AD8232/ADS1115/OLED BOM was replaced. Earlier pipeline/model code and its tests remain at stable paths to preserve reproducibility and imports. Historical narrative documents are marked as such. The earlier seed-7 model bundle is historical, not the deployment artifact.

Removed an imported pre-correction backup script, an empty artifact placeholder and KiCad personal UI state. Original archives retain source provenance. No patient recordings or personal project handover text were copied into public documentation. Three small export fixtures were retained with the supplied deployment package.

No new training, eligibility selection, threshold selection or held-out model tuning was performed. Verification results are recorded separately in VALIDATION.md. Remaining reproduction gaps are explicit in [PROJECT_STATE.md](../PROJECT_STATE.md).

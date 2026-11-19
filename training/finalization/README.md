# AFib finalization fast package

This package freezes the selected 20-minute forecasting recipe and trains a
separate patient-disjoint current-AF safety detector from the existing Icentia
30-second morphology cache. It never reads CPSC2021 or SHDB-AF.

The current-AF detector is a safety component, not evidence that the forecasting
model predicts AF reliably. External validation and Raspberry Pi measurements
remain separate required phases.

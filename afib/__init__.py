"""Causal, patient-separated AF onset prediction."""

FS = 250
OBS_SECONDS = 600
HORIZON_SECONDS = 1200
HRV_ORDER = ("RMSSD", "SDNN", "LF/HF", "Mean RR", "pNN50", "Sample Entropy")

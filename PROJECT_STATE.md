# Project state — 20 November 2026

The deployed forecast is frozen: GroupNorm/curriculum SE-CNN-HRV-UniLSTM, 159,503 parameters, fold 0, threshold 0.3310546875. See [model identities](docs/FINAL_MODEL.md) and [raw results](results).

| Component | Evidence/status |
| --- | --- |
| Frozen model, HRV state, ONNX | Exact imported bytes; SHA-256 identities in release manifest |
| Validation and sealed test | Original summaries and predictions included; no retraining or threshold tuning in this update |
| Later fivefold development | Original aggregate and BCE fold files; separate experiment |
| Wi-Fi replay | Prior recorded Pi runs: POS102 0.35684036; NEG024 0.05910458, cutoff 0.3310546875 |
| Live acquisition | Waveform receiver and ESP32 source included; subsequent successful use reported by the team |
| Android | Latest supplied v1.4.8 live-resume source; build/device behavior needs Android Studio verification |
| PCB | Supplied Rev B fabrication candidate and extracted source; prior manufacture/bench bring-up reported |
| Benchmarks | Final Pi latency, peak RAM, battery life and long-duration reliability measurements still needed |

## Reproduction gaps

- Complete training caches, unified/harmonized manifests and locked patient splits are absent from supplied final archives. Their original hashes remain in `FORECAST_FREEZE.json`. Recover the exact files before claiming full training reproduction.
- The later SHDB development manifest and missing-record decisions need recovery. SHDB must not be silently attributed to the earlier frozen checkpoint.
- POS102.BIN/NEG024.BIN are expected on the project's SD card/capture path; they are not included. Receivers enforce exact byte counts and hashes.
- The deployed ESP32 Wi-Fi SD replay firmware was not found in recovered source files. The included live sketch cannot substitute for it.
- A measured ten-minute live inference log, final locked Pi versions, final Android build verification and cost reconciliation remain to be supplied.

Use the root README and final model/results documents for current claims. Preserve all frozen settings and patient separation; never select settings using the sealed test.

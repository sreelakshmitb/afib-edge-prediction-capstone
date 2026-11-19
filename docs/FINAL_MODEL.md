# Frozen forecast model

Final recipe: `groupnorm_curriculum_corrected_evaluation`, SE-CNN-HRV-UniLSTM, 159,503 parameters. Three convolution stages (16/32/64 channels, kernels 7/7/5), GroupNorm, SE gates, 128-dimensional segment embedding, 128-unit unidirectional LSTM, last-state classification. See [model implementation](../training/unified/src/afib_unified/models.py) and [training recipe](../training/unified/src/controlled_repair.py).

| Artifact | SHA-256 |
| --- | --- |
| `forecast_checkpoint.pt` | `7444dc2391e1de5edd4dbd42f4e1515bb6462f3ea14ed25b847bc7004e90003a` |
| `forecast_model.onnx` | `deffa6242d1d411472c9505ff9fdb416d60a047d0f7cad347508a16d3408d4ed` |
| v1.7 corrected-eligibility training archive | `4b6b131e3b076ccc11b53b25b084c57ade1eedc7e574c95b8d2ce9c2abf5e667` |

## Contract and preprocessing

250 Hz, ten-minute observation, twenty sequential 30-second segments. ECG float32 `[B,20,1,7500]`; HRV float32 `[B,20,6]`. HRV order: RMSSD (ms), SDNN (ms), LF/HF, mean RR (ms), pNN50 (%), sample entropy. Export fixtures use batch 1. The graph returns a logit: apply sigmoid externally and compare with **0.3310546875**, chosen on validation.

Replay converts int32 microvolts to millivolts, applies the original band-pass, XQRS, per-segment HRV, per-segment ECG z-score and saved training-only HRV imputation/scaling. The graph does not implement raw acquisition/filtering/HRV extraction internally.

The task is AF onset within twenty minutes after an eligible non-AF observation, not exact onset timing or current-AF detection. Live collection alone does not establish rhythm eligibility; this remains a deployment limitation.

## Loading and parity

```bash
python -m pip install torch numpy scipy scikit-learn pandas onnxruntime
python scripts/verify_frozen_model.py --checkpoint
```

The script checks artifact identities, builds `build_model("proposed-groupnorm")`, loads `model_state_dict` strictly, counts parameters and compares checkpoint/ONNX fixture outputs. The trusted checkpoint includes optimizer/RNG state and requires `weights_only=False`; the script checks its exact SHA-256 first.

Recorded real-window export probability differences were below 5e-8. Metadata records synthetic-input maximum logit difference 1.9073486328125e-6. This establishes numerical parity, not clinical accuracy or Pi speed.

The primary test is one held-out patient fold with 23.31% recall. See [results](RESULTS.md). No CPSC2021 score exists under the strict task gate. Auxiliary transfer/ablation models and older seed-7 inner checkpoints did not replace this artifact. Preserve all freeze settings and patient splits.

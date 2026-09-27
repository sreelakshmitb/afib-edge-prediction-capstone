# AFib 20-Minute Prediction Model Bundle

## Selected prediction model

- Candidate: `se_last_mask`
- Architecture: SE-CNN-HRV-UniLSTM
- Trainable parameters: 159,503
- Observation window: 10 minutes
- Segments: 20 chronological segments of 30 seconds
- Prediction horizon: approximately 20 minutes before AF onset
- Sampling rate: 250 Hz
- ECG input shape: [B, 20, 1, 7500]
- HRV input shape: [B, 20, 6]

## Development-validation performance

- Mean five-fold validation AP: 0.182597
- Pooled out-of-fold AP: 0.153119
- Pooled out-of-fold AUROC: 0.599308
- Mean within-record AUROC: 0.575418

## Checkpoint interpretation

This bundle contains five patient/record-disjoint inner-validation
checkpoints. Each checkpoint was trained with a different development
split. They are cross-validation checkpoints, not five independent
final models and not one model trained on the complete dataset.

Do not report the highest-performing individual fold as the overall
performance. Use the aggregate metrics above.

The experimental AP-ranking variant did not pass the pre-specified
robustness gate, so `se_last_mask` remains the selected prediction model.

No raw ECG data or patient signals are included.

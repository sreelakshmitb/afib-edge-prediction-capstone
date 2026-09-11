# Paper Reproducibility Audit
Inspected supplied five-page IEEE paper, DOI 10.1109/ICASSP55912.2026.11464723,
including Figure 1 visually. The paper is not copied into Git.

Confirmed: 10-minute normal-rhythm observation, twenty 30-second segments,
20-minute AF horizon, 0.5–40 Hz filtering, three CNN layers with kernels 5–7,
BN/ReLU, 128-dimensional CNN embedding plus six HRV features, single-layer
unidirectional LSTM. Figure 1 specifies 128 hidden states and mean pooling.
Training section gives AdamW, learning rate 0.0005, weight decay 0.0001,
dropout 0.3, gradient clipping 1.0 and a fixed decision threshold of 0.5.

Unspecified or inconsistent: exact CNN widths, strides and pooling; complete
HRV feature definitions; R-peak method and correction; focal-loss parameters;
masking details; label sampling and inner split details. Text refers to final
hidden states whereas Figure 1 shows mean pooling. Figure 1 scales all hybrid
features but does not explain how the scaler handles a learned CNN embedding.
AFDB's two annotation-only records cannot provide ECG despite the paper's
109-record combined-corpus count. Some external-dataset rates/counts and metric
values are inconsistent. Reported paper metrics are not our results.

Implementation decisions: CNN widths 32/64/128, kernels 7/5/5, stride 5 each,
global average pooling, UniLSTM hidden 128, temporal mean pooling per Figure 1.
HRV scaler is fit only on inner-training patients; CNN BN statistics are learned
only during training. No separately fitted scaler over changing CNN embeddings.
Primary model adds SE after each CNN block, reduction ratio four. Both retain
the same 128+6 representation and unidirectional recurrence.

Initial experiment deviations: AFDB only, fixed non-overlapping observations,
no class downsampling, weighted BCE rather than combined focal loss and weighted
sampling, no random masking, and validation BCE for checkpoint selection.
These are explicitly a paper-aligned implementation, not an exact reproduction.
The project's fixed HRV order overrides the paper's incomplete HRV description.
See PREPROCESSING_PROTOCOL.md for the declared causal HRV and exclusion rules.

# Checkpoint Loading Notes

1. Install the dependencies specified in `code/pyproject.toml`.
2. Add `code/src` to the Python path.
3. Construct the same `se_last_mask` SE-CNN-HRV-UniLSTM architecture
   using the included source code.
4. Load one checkpoint from the `checkpoints/` directory.
5. Inspect the checkpoint dictionary keys before selecting the stored
   model state dictionary.
6. Use ECG tensors shaped [B, 20, 1, 7500] and HRV tensors shaped
   [B, 20, 6].

The five checkpoints correspond to inner folds 0 through 4.

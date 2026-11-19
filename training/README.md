# Training sources and reproduction

- `unified/`: extracted v1.7 corrected-eligibility training source associated with the frozen checkpoint.
- `archives/`: exact original v1.7 ZIP; SHA-256 is recorded in the freeze manifest.
- `finalization/`: supplied fast finalization source and its original embedded training archive. It also contains historical current-AF/dual-system experiments; these are not the deployed forecast.
- [notebooks](../notebooks): supplied audit/export and prediction-audit notebooks, with outputs cleared. They are historical audit workflows, not proof of a fresh run.
- [old source bundle](../model_bundles/se_last_mask_seed7): earlier training/pretraining and signal/HRV code; historical checkpoints remain distinguishable.

## Frozen checkpoint source

```bash
python -m pip install torch numpy scipy pandas scikit-learn wfdb pytest
export PYTHONPATH="$PWD/training/unified/src"
python training/unified/src/controlled_repair.py --help
python scripts/verify_frozen_model.py --checkpoint
```

On PowerShell set `$env:PYTHONPATH = "$PWD/training/unified/src"`. The selected architecture is `proposed-groupnorm`; the saved checkpoint recipe is read from the artifact, not inferred from the filename. Use its exact state dictionary and saved HRV preprocessing.

Recover the original unified/harmonized cache manifests, locked patient splits and caches from the paths and hashes in [FORECAST_FREEZE.json](../artifacts/frozen_forecast_v1/FORECAST_FREEZE.json). The original recipe summary supplies model/sampler/curriculum/eligibility settings. Source alone cannot reconstruct missing data or lock files. Do not rerun training with a random split and call it reproduction.

GroupKFold(5) and training-only preprocessing statistics remain fixed. The later SHDB-assisted fivefold BCE development run is a separate experiment with a different manifest hash. Its complete execution source/checkpoints were not supplied in the final result archive, so those results are evidence imports, not a fully runnable reconstruction.

The current export notebook may refer to Colab/Drive paths. Update paths for your environment, verify frozen identities, and retain all patient-separation rules. The public Colab supplied by the team is [here](https://colab.research.google.com/drive/1b_21karcCI4RHoNMcGsVOIjKHKb6YWs1); it was not executed during this update.

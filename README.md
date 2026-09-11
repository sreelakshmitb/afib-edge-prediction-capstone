# afib-edge-prediction-capstone
Lightweight short-term atrial fibrillation prediction from single-lead ECG using PyTorch, HRV, SE-CNN-UniLSTM and Raspberry Pi 4B deployment.

The software pipeline now supports real AFDB preprocessing, patient-separated
training, and FP32 ONNX export. Experimental status is recorded in PROJECT_STATE.md;
implemented code alone is not evidence of trained-model performance.

Tested environment: Windows, Python 3.14.6, CPU PyTorch 2.14.0. Install the pinned
`requirements-lock.txt` into a project virtual environment. The initial experiment
root is `C:/Users/Lenovo/afib-edge-data`, outside this repository.

```powershell
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/python.exe -u -m afib.preprocess --root C:/Users/Lenovo/afib-edge-data --records all --workers 2
.venv/Scripts/python.exe -m afib.audit --root C:/Users/Lenovo/afib-edge-data
.venv/Scripts/python.exe -u -m afib.train --root C:/Users/Lenovo/afib-edge-data --run smoke --model se --epochs 1 --max-steps 2
.venv/Scripts/python.exe -u -m afib.train --root C:/Users/Lenovo/afib-edge-data --run initial --model se --folds 0
```

Run names must be new: existing fold directories are never silently overwritten.
Smoke training never evaluates outer-test outcomes. Full training selects an epoch
using inner-validation BCE, then evaluates its outer-test patients once at threshold
0.5. The final deployed architecture is UNDECIDED. Do not select an architecture
or checkpoint by comparing outer-test scores. Export support is implemented;
it does not imply a deployment decision or a successfully exported trained model.

CNN-HRV-UniLSTM is the PAPER REPRODUCTION / REFERENCE BASELINE, not an original
capstone contribution. Unspecified paper details are REPRODUCTION CHOICE items
in the reproducibility audit. SE-CNN-HRV-UniLSTM adds a CAPSTONE EXTENSION and
remains a proposed candidate. Both candidates have fewer than 250,000 parameters.

After an interrupted download, resume only unfinished record IDs with `--records`.
Once all per-record outputs exist, `python -m afib.manifest --root ...` assembles
them without recomputing completed records and checks source and code hashes.
Then run the independent `afib.audit` check before training.

See `docs/PREPROCESSING_PROTOCOL.md` for labels, feature definitions and exclusions,
and `docs/PAPER_REPRODUCIBILITY_AUDIT.md` for paper alignment and explicit deviations.
The exported graph includes HRV imputation/scaling but expects already filtered,
observation-normalized ECG. It does not implement raw ECG acquisition or R peaks.

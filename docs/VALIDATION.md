# Repository update validation

The update preserves the exact frozen checkpoint, ONNX graph, preprocessing JSON and original result files. No training, threshold tuning or new held-out evaluation was performed.

- `python scripts/verify_release.py`: passed, 21 frozen evidence files; prediction CSV reproduces TP 55, FN 181, FP 142, TN 1,431.
- Python syntax compilation: passed for original pipeline, deployment scripts, verification/setup scripts and imported training/finalization source.
- `python scripts/prepare_pi.py`: passed, assembled the expected standalone `_training_src` import layout without changing the graph or HRV state.
- Local ONNX execution: passed on all three supplied fixtures; probabilities match the saved ONNX values exactly (0.10818697626809322, 0.1477449035144997, 0.11379623794400998).
- Validation/test prediction files have no shared subject IDs. Notebook JSON, Android XML and current-document relative links passed structural checks.
- GitHub CI passed checkpoint reconstruction, all three ONNX fixtures, existing contract tests and corrected-training package tests for the initial release commit (`977aaf5`). [Run evidence](https://github.com/sreelakshmitb/afib-edge-prediction-capstone/actions/runs/44594386234).

See the pull request's CI status for the actual remote run outcome. Android/ESP32 builds, physical Pi replay/live acquisition, PCB manufacture, long-duration stability, power and clinical validation were not rerun in this repository update. The original working device evidence and the team's later reported operation remain separate from software checks.

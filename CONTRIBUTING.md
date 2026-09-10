# Contributing

This repository is maintained by a three-member capstone team.

## Branch Policy

Do not develop directly on `main`.

Create a branch for each task using names such as:

- `feature/data-pipeline`
- `feature/model`
- `feature/training`
- `feature/onnx-export`
- `feature/pi-acquisition`
- `docs/thesis`
- `fix/<short-description>`

## Workflow

1. Pull the latest `main`.
2. Create a task branch.
3. Make focused changes.
4. Run relevant tests.
5. Commit with a descriptive message.
6. Push the branch.
7. Open a Pull Request into `main`.
8. Another team member reviews the PR.
9. Merge only after review and required tests pass.
10. Everyone updates their local `main`.

## Commit Style

Examples:

- `feat: add AFDB record loader`
- `feat: implement subject-wise GroupKFold`
- `test: add patient leakage assertions`
- `fix: prevent validation scaler leakage`
- `docs: update preprocessing protocol`

## Important Rules

- Never commit passwords, API keys, tokens, or credentials.
- Never commit raw ECG datasets.
- Never commit virtual environments.
- Never fabricate experiment results.
- Preserve patient-wise leakage prevention.
- Preserve the fixed tensor and HRV contracts.
- Keep the deployed model below 250,000 parameters.
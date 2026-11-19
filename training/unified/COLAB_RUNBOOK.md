# Colab runbook

## Current next step (v1.6): harmonized legacy negative timing

Run section 1 after a fresh runtime, then these cells. The annotation step
reads no ECG signal, and the training step reuses the immutable caches.

### A. Build the cache-preserving timing correction

```python
from google.colab import drive
from pathlib import Path
import json, os, subprocess, sys

drive.mount("/content/drive")
root = Path("/content/drive/MyDrive/AFib_Capstone")
project = Path("/content/afib-unified-training-v1")
output = root / "manifests/harmonized_legacy_timing_v1"
env = {**os.environ, "PYTHONPATH": str(project / "src")}
subprocess.run([
    sys.executable, str(project / "src/enrich_legacy_timing.py"),
    "--target-dir", str(root / "manifests/multihorizon_v1"),
    "--protocol-dir", str(root / "manifests/unified_training_v1"),
    "--afdb-root", "/content/drive/MyDrive/af_proj/init_model/data/afdb",
    "--ltafdb-root", "/content/drive/MyDrive/af_proj/data/ltafdb",
    "--output-dir", str(output),
], check=True, env=env)
summary = json.loads((output / "harmonized_summary.json").read_text())
assert summary["all_gates_pass"]
assert summary["outer_test_evaluated"] is False
```

### B. Run the one-variable controlled repair

```python
from pathlib import Path
import json, os, subprocess, sys

root = Path("/content/drive/MyDrive/AFib_Capstone")
project = Path("/content/afib-unified-training-v1")
stage = Path("/content/afib_unified_stage_v1")
harmonized = root / "manifests/harmonized_legacy_timing_v1"
output = root / "runs/unified_harmonized_negatives_v1/fold0"
output.mkdir(parents=True, exist_ok=True)
env = {**os.environ, "PYTHONPATH": str(project / "src")}
if not (stage / "STAGE_COMPLETE.json").is_file():
    subprocess.run([
        sys.executable, str(project / "src/stage_caches.py"),
        "--legacy-source", str(root / "cache/v2_chunked_xqrs"),
        "--iridia-source", str(root / "cache/iridia_signal_cache_v1"),
        "--protocol-dir", str(root / "manifests/unified_training_v1"),
        "--stage-root", str(stage),
    ], check=True, env=env)
command = [
    sys.executable, str(project / "src/controlled_repair.py"),
    "--legacy-cache", str(stage / "legacy"),
    "--iridia-cache", str(stage / "iridia"),
    "--protocol-dir", str(root / "manifests/unified_training_v1"),
    "--manifest-path", str(harmonized / "harmonized_cached_manifest.csv"),
    "--manifest-summary", str(harmonized / "harmonized_summary.json"),
    "--output-dir", str(output),
    "--recipe", "groupnorm-curriculum-harmonized-negatives",
    "--fold", "0", "--epochs", "18", "--patience", "5",
    "--batches-per-epoch", "50", "--resume",
]
log_path = output / "training.log"
with log_path.open("a", encoding="utf-8") as log:
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, env=env,
    )
    for line in process.stdout:
        print(line, end="")
        log.write(line)
        log.flush()
    return_code = process.wait()
if return_code:
    raise RuntimeError(f"harmonized repair failed: {log_path}")
summary = json.loads((output / "controlled_repair_summary.json").read_text())
assert summary["all_gates_pass"]
assert summary["outer_test_evaluated"] is False
```

### C. Compare on identical corrected validation rows

```python
from pathlib import Path
import json, os, subprocess, sys

root = Path("/content/drive/MyDrive/AFib_Capstone")
project = Path("/content/afib-unified-training-v1")
candidate = root / "runs/unified_harmonized_negatives_v1/fold0"
output = candidate / "harmonized_comparison.json"
env = {**os.environ, "PYTHONPATH": str(project / "src")}
subprocess.run([
    sys.executable, str(project / "src/compare_harmonized_negatives.py"),
    "--baseline-run", str(root / "runs/unified_controlled_repairs_v1/fold0/groupnorm_curriculum"),
    "--candidate-run", str(candidate),
    "--harmonized-dir", str(root / "manifests/harmonized_legacy_timing_v1"),
    "--output", str(output),
], check=True, env=env)
summary = json.loads(output.read_text())
assert summary["outer_test_unlock"] is False
assert summary["outer_test_evaluated"] is False
```

## Current next step (v1.5): multi-horizon structural reconstruction

After section 1 succeeds, use the three cells below. They reuse the existing
local SSD stage and never construct an outer-test loader.

### A. Reconstruct and lock exact five-horizon targets

```python
from google.colab import drive
from pathlib import Path
import json, os, subprocess, sys

drive.mount("/content/drive")
root = Path("/content/drive/MyDrive/AFib_Capstone")
project = Path("/content/afib-unified-training-v1")
protocol = root / "manifests/unified_training_v1"
targets = root / "manifests/multihorizon_v1"
env = {**os.environ, "PYTHONPATH": str(project / "src")}
subprocess.run([
    sys.executable, str(project / "src/build_multihorizon_targets.py"),
    "--cached-manifest", str(protocol / "unified_cached_manifest.csv"),
    "--combined-manifest", str(root / "manifests/balanced_training_v1/combined_sampling_manifest.csv"),
    "--output-dir", str(targets),
], check=True, env=env)
summary = json.loads((targets / "target_summary.json").read_text())
assert summary["all_gates_pass"]
assert summary["outer_test_evaluated"] is False
```

### B. Train the resumable fold-0 multi-horizon model

```python
from pathlib import Path
import json, os, subprocess, sys

root = Path("/content/drive/MyDrive/AFib_Capstone")
project = Path("/content/afib-unified-training-v1")
stage = Path("/content/afib_unified_stage_v1")
targets = root / "manifests/multihorizon_v1"
output = root / "runs/unified_multihorizon_v1/fold0"
output.mkdir(parents=True, exist_ok=True)
env = {**os.environ, "PYTHONPATH": str(project / "src")}
if not (stage / "STAGE_COMPLETE.json").is_file():
    print("Local SSD stage is missing; rebuilding it from the immutable Drive caches...")
    subprocess.run([
        sys.executable, str(project / "src/stage_caches.py"),
        "--legacy-source", str(root / "cache/v2_chunked_xqrs"),
        "--iridia-source", str(root / "cache/iridia_signal_cache_v1"),
        "--protocol-dir", str(root / "manifests/unified_training_v1"),
        "--stage-root", str(stage),
    ], check=True, env=env)
command = [
    sys.executable, str(project / "src/multihorizon_validation.py"),
    "--legacy-cache", str(stage / "legacy"),
    "--iridia-cache", str(stage / "iridia"),
    "--protocol-dir", str(root / "manifests/unified_training_v1"),
    "--target-dir", str(targets), "--output-dir", str(output),
    "--fold", "0", "--epochs", "18", "--patience", "5",
    "--batches-per-epoch", "50", "--resume",
]
log_path = output / "training.log"
with log_path.open("a", encoding="utf-8") as log:
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, env=env,
    )
    for line in process.stdout:
        print(line, end="")
        log.write(line)
        log.flush()
    return_code = process.wait()
if return_code:
    raise RuntimeError(f"multi-horizon training failed: {log_path}")
summary = json.loads((output / "multihorizon_summary.json").read_text())
assert summary["all_gates_pass"]
assert summary["outer_test_evaluated"] is False
```

### C. Compare with the frozen single-head model

```python
from pathlib import Path
import json, os, subprocess, sys

root = Path("/content/drive/MyDrive/AFib_Capstone")
project = Path("/content/afib-unified-training-v1")
candidate = root / "runs/unified_multihorizon_v1/fold0"
output = candidate / "structural_comparison.json"
env = {**os.environ, "PYTHONPATH": str(project / "src")}
subprocess.run([
    sys.executable, str(project / "src/compare_multihorizon.py"),
    "--baseline-run", str(root / "runs/unified_controlled_repairs_v1/fold0/groupnorm_curriculum"),
    "--multihorizon-run", str(candidate),
    "--target-dir", str(root / "manifests/multihorizon_v1"),
    "--output", str(output),
], check=True, env=env)
summary = json.loads(output.read_text())
print(json.dumps(summary, indent=2))
assert summary["outer_test_unlock"] is False
assert summary["outer_test_evaluated"] is False
```

## 1. Upload, install, and test

```python
from google.colab import files
from pathlib import Path
import os, shutil, subprocess, sys, zipfile

uploaded = files.upload()
archives = [Path("/content") / name for name in uploaded if name.lower().endswith(".zip")]
assert len(archives) == 1, f"Upload exactly one ZIP; found {archives}"
project = Path("/content/afib-unified-training-v1")
if project.exists():
    shutil.rmtree(project)
with zipfile.ZipFile(archives[0]) as bundle:
    roots = {Path(name).parts[0] for name in bundle.namelist() if Path(name).parts}
    assert "afib-unified-training-v1" in roots, f"Unexpected ZIP roots: {sorted(roots)}"
    bundle.extractall("/content")
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "-r", str(project / "requirements.txt")],
    check=True,
)
env = {**os.environ, "PYTHONPATH": str(project / "src")}
subprocess.run(
    [sys.executable, "-m", "pytest", "-q", str(project / "tests")],
    cwd=project, env=env, check=True,
)
print("UNIFIED TRAINING PACKAGE READY")
```

## 2. Freeze the protocol before training

```python
from google.colab import drive
from pathlib import Path
import json, os, subprocess, sys

drive.mount("/content/drive")
root = Path("/content/drive/MyDrive/AFib_Capstone")
project = Path("/content/afib-unified-training-v1")
protocol = root / "manifests/unified_training_v1"
env = {**os.environ, "PYTHONPATH": str(project / "src")}
subprocess.run([
    sys.executable, str(project / "src/build_protocol.py"),
    "--combined-manifest", str(root / "manifests/balanced_training_v1/combined_sampling_manifest.csv"),
    "--iridia-plan", str(root / "manifests/iridia_signal_cache_plan_v1/extraction_plan.csv"),
    "--output-dir", str(protocol), "--seed", "7",
], check=True, env=env)
summary = json.loads((protocol / "protocol_summary.json").read_text())
splits = json.loads((protocol / "locked_patient_splits.json").read_text())
print("\nLOCKED UNIFIED TRAINING PROTOCOL")
print("=" * 60)
for key in ("all_gates_pass", "windows", "positives", "negatives", "patients"):
    print(f"{key}: {summary[key]}")
print("source_windows:", summary["source_windows"])
print("source_positives:", summary["source_positives"])
for fold in splits["folds"]:
    print(
        f"fold {fold['fold']}: "
        f"train={fold['train']['windows']}w/{fold['train']['patients']}p; "
        f"val={fold['validation']['windows']}w/{fold['validation']['patients']}p; "
        f"test={fold['test']['windows']}w/{fold['test']['patients']}p"
    )
assert summary["all_gates_pass"]
```

## 3. One balanced GPU forward/backward batch

```python
from pathlib import Path
import json, os, subprocess, sys

root = Path("/content/drive/MyDrive/AFib_Capstone")
project = Path("/content/afib-unified-training-v1")
output = root / "runs/unified_training_v1_gpu_smoke"
env = {**os.environ, "PYTHONPATH": str(project / "src")}
subprocess.run([
    sys.executable, str(project / "src/gpu_smoke.py"),
    "--legacy-cache", str(root / "cache/v2_chunked_xqrs"),
    "--iridia-cache", str(root / "cache/iridia_signal_cache_v1"),
    "--protocol-dir", str(root / "manifests/unified_training_v1"),
    "--output-dir", str(output), "--fold", "0",
], check=True, env=env)
summary = json.loads((output / "gpu_smoke_summary.json").read_text())
print("\nUNIFIED GPU SMOKE")
print("=" * 60)
for key in (
    "all_gates_pass", "device", "parameter_count", "loss", "gradient_norm",
    "peak_gpu_memory_mib", "train_windows_used_for_hrv_fit", "batch_positive_count",
):
    print(f"{key}: {summary[key]}")
print("source_by_label:", summary["source_by_label"])
print("positive_lead_bins:", summary["positive_lead_bins"])
print("gates:")
for key, value in summary["gates"].items():
    print(f"  {key}: {value}")
assert summary["all_gates_pass"]
```

Stop after the GPU smoke and preserve its complete output. Full training is a
separate gated package so a smoke failure cannot silently start a long run.

## 4. Stage immutable caches to local SSD

```python
from pathlib import Path
import json, os, subprocess, sys

root = Path("/content/drive/MyDrive/AFib_Capstone")
project = Path("/content/afib-unified-training-v1")
stage = Path("/content/afib_unified_stage_v1")
env = {**os.environ, "PYTHONPATH": str(project / "src")}
subprocess.run([
    sys.executable, str(project / "src/stage_caches.py"),
    "--legacy-source", str(root / "cache/v2_chunked_xqrs"),
    "--iridia-source", str(root / "cache/iridia_signal_cache_v1"),
    "--protocol-dir", str(root / "manifests/unified_training_v1"),
    "--stage-root", str(stage),
], check=True, env=env)
summary = json.loads((stage / "STAGE_COMPLETE.json").read_text())
print("\nLOCAL TRAINING STAGE")
print("=" * 60)
print("all_gates_pass:", summary["all_gates_pass"])
print("records:", summary["records"])
print("files:", summary["files"])
print("cache_GiB:", round(summary["required_bytes"] / 1024**3, 3))
print("newly_copied_GiB:", round(summary["newly_copied_bytes"] / 1024**3, 3))
assert summary["all_gates_pass"]
```

## 5. Resumable validation-only fold-0 pilot

```python
from pathlib import Path
import json, os, subprocess, sys

root = Path("/content/drive/MyDrive/AFib_Capstone")
project = Path("/content/afib-unified-training-v1")
stage = Path("/content/afib_unified_stage_v1")
output = root / "runs/unified_validation_pilot_v1/fold0"
env = {**os.environ, "PYTHONPATH": str(project / "src")}
subprocess.run([
    sys.executable, str(project / "src/validation_pilot.py"),
    "--legacy-cache", str(stage / "legacy"),
    "--iridia-cache", str(stage / "iridia"),
    "--protocol-dir", str(root / "manifests/unified_training_v1"),
    "--output-dir", str(output),
    "--fold", "0", "--epochs", "15", "--patience", "5",
    "--batches-per-epoch", "50", "--resume",
], check=True, env=env)
summary = json.loads((output / "validation_pilot_summary.json").read_text())
print("\nVALIDATION-ONLY PILOT")
print("=" * 60)
for key in ("all_gates_pass", "device", "parameter_count", "train_windows", "validation_windows", "best_epoch", "outer_test_evaluated"):
    print(f"{key}: {summary[key]}")
print("validation_ranking:", summary["validation_ranking"])
print("validation_operating_point:", summary["validation_operating_point"])
print("validation_at_0_5:", summary["validation_at_0_5"])
print("validation_by_source:", summary["validation_by_source"])
print("recall_by_lead_bin:", summary["validation_positive_recall_by_lead_bin"])
print("gates:", summary["gates"])
assert summary["all_gates_pass"]
assert summary["outer_test_evaluated"] is False
```

## 10. Conditional domain-alignment repair

```python
from pathlib import Path
import os, subprocess, sys

root = Path("/content/drive/MyDrive/AFib_Capstone")
project = Path("/content/afib-unified-training-v1")
stage = Path("/content/afib_unified_stage_v1")
runs_root = root / "runs/unified_controlled_repairs_v1/fold0"
recipe = "groupnorm-curriculum-domain-align"
output = runs_root / recipe.replace("-", "_")
output.mkdir(parents=True, exist_ok=True)
log_path = output / "training.log"
env = {**os.environ, "PYTHONPATH": str(project / "src")}
command = [
    sys.executable, str(project / "src/controlled_repair.py"),
    "--legacy-cache", str(stage / "legacy"),
    "--iridia-cache", str(stage / "iridia"),
    "--protocol-dir", str(root / "manifests/unified_training_v1"),
    "--output-dir", str(output), "--recipe", recipe,
    "--fold", "0", "--epochs", "18", "--patience", "5",
    "--batches-per-epoch", "50", "--resume",
]
with log_path.open("a", encoding="utf-8") as log:
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, env=env,
    )
    for line in process.stdout:
        print(line, end="")
        log.write(line)
        log.flush()
    return_code = process.wait()
if return_code:
    raise RuntimeError(f"domain-alignment repair failed: {log_path}")
print("CONDITIONAL DOMAIN-ALIGNMENT TRAINING COMPLETE")
```

## 11. Compare domain alignment with full fusion

```python
from pathlib import Path
import json, os, subprocess, sys

root = Path("/content/drive/MyDrive/AFib_Capstone")
project = Path("/content/afib-unified-training-v1")
runs_root = root / "runs/unified_controlled_repairs_v1/fold0"
output = runs_root / "domain_alignment_comparison.json"
env = {**os.environ, "PYTHONPATH": str(project / "src")}
subprocess.run([
    sys.executable, str(project / "src/compare_domain_alignment.py"),
    "--runs-root", str(runs_root),
    "--protocol-dir", str(root / "manifests/unified_training_v1"),
    "--output", str(output),
], check=True, env=env)
summary = json.loads(output.read_text())
assert summary["outer_test_unlock"] is False
assert summary["outer_test_evaluated"] is False
```

## 6. Run the two predeclared validation-only repairs

```python
from pathlib import Path
import json, os, subprocess, sys

root = Path("/content/drive/MyDrive/AFib_Capstone")
project = Path("/content/afib-unified-training-v1")
stage = Path("/content/afib_unified_stage_v1")
repairs_root = root / "runs/unified_controlled_repairs_v1/fold0"
env = {**os.environ, "PYTHONPATH": str(project / "src")}

# This is resume-safe and copies only missing or size-mismatched immutable files.
subprocess.run([
    sys.executable, str(project / "src/stage_caches.py"),
    "--legacy-source", str(root / "cache/v2_chunked_xqrs"),
    "--iridia-source", str(root / "cache/iridia_signal_cache_v1"),
    "--protocol-dir", str(root / "manifests/unified_training_v1"),
    "--stage-root", str(stage),
], check=True, env=env)

for recipe, epochs in (("groupnorm", 15), ("groupnorm-curriculum", 18)):
    output = repairs_root / recipe.replace("-", "_")
    output.mkdir(parents=True, exist_ok=True)
    log_path = output / "training.log"
    command = [
        sys.executable, str(project / "src/controlled_repair.py"),
        "--legacy-cache", str(stage / "legacy"),
        "--iridia-cache", str(stage / "iridia"),
        "--protocol-dir", str(root / "manifests/unified_training_v1"),
        "--output-dir", str(output),
        "--recipe", recipe,
        "--fold", "0", "--epochs", str(epochs), "--patience", "5",
        "--batches-per-epoch", "50", "--resume",
    ]
    print(f"\nSTARTING {recipe.upper()} VALIDATION REPAIR", flush=True)
    with log_path.open("a", encoding="utf-8") as log:
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, env=env,
        )
        for line in process.stdout:
            print(line, end="")
            log.write(line)
            log.flush()
        return_code = process.wait()
    if return_code:
        raise RuntimeError(f"{recipe} failed with return code {return_code}")

print("BOTH CONTROLLED VALIDATION REPAIRS COMPLETE")
```

## 7. Apply the locked progression gate

```python
from pathlib import Path
import json, os, subprocess, sys

root = Path("/content/drive/MyDrive/AFib_Capstone")
project = Path("/content/afib-unified-training-v1")
repairs_root = root / "runs/unified_controlled_repairs_v1/fold0"
comparison_path = repairs_root / "comparison_summary.json"
env = {**os.environ, "PYTHONPATH": str(project / "src")}

subprocess.run([
    sys.executable, str(project / "src/compare_repairs.py"),
    "--baseline-predictions",
    str(root / "runs/unified_validation_pilot_v1/fold0/best_validation_predictions.csv"),
    "--repairs-root", str(repairs_root),
    "--protocol-dir", str(root / "manifests/unified_training_v1"),
    "--output", str(comparison_path),
], check=True, env=env)

comparison = json.loads(comparison_path.read_text())
print("\nFINAL VALIDATION DECISION")
print("=" * 72)
print("selected_candidate:", comparison["selected_candidate"])
print("outer_test_unlock:", comparison["outer_test_unlock"])
print("decision:", comparison["decision"])
print("outer_test_evaluated:", comparison["outer_test_evaluated"])
assert comparison["outer_test_evaluated"] is False
```

## 8. Independently train ECG-only and HRV-only ablations

```python
from pathlib import Path
import os, subprocess, sys

root = Path("/content/drive/MyDrive/AFib_Capstone")
project = Path("/content/afib-unified-training-v1")
stage = Path("/content/afib_unified_stage_v1")
runs_root = root / "runs/unified_controlled_repairs_v1/fold0"
env = {**os.environ, "PYTHONPATH": str(project / "src")}

experiments = (
    "groupnorm-curriculum-ecg-only",
    "groupnorm-curriculum-hrv-only",
)
for recipe in experiments:
    output = runs_root / recipe.replace("-", "_")
    output.mkdir(parents=True, exist_ok=True)
    log_path = output / "training.log"
    command = [
        sys.executable, str(project / "src/controlled_repair.py"),
        "--legacy-cache", str(stage / "legacy"),
        "--iridia-cache", str(stage / "iridia"),
        "--protocol-dir", str(root / "manifests/unified_training_v1"),
        "--output-dir", str(output), "--recipe", recipe,
        "--fold", "0", "--epochs", "18", "--patience", "5",
        "--batches-per-epoch", "50", "--resume",
    ]
    print(f"\nSTARTING {recipe.upper()}", flush=True)
    with log_path.open("a", encoding="utf-8") as log:
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, env=env,
        )
        for line in process.stdout:
            print(line, end="")
            log.write(line)
            log.flush()
        return_code = process.wait()
    if return_code:
        raise RuntimeError(f"{recipe} failed with return code {return_code}")

print("INDEPENDENT BRANCH TRAINING COMPLETE")
```

## 9. Compare the independently trained branches

```python
from pathlib import Path
import json, os, subprocess, sys

root = Path("/content/drive/MyDrive/AFib_Capstone")
project = Path("/content/afib-unified-training-v1")
runs_root = root / "runs/unified_controlled_repairs_v1/fold0"
output = runs_root / "branch_ablation_summary.json"
env = {**os.environ, "PYTHONPATH": str(project / "src")}

subprocess.run([
    sys.executable, str(project / "src/compare_branch_ablations.py"),
    "--runs-root", str(runs_root),
    "--protocol-dir", str(root / "manifests/unified_training_v1"),
    "--output", str(output),
], check=True, env=env)

summary = json.loads(output.read_text())
assert summary["outer_test_unlock"] is False
assert summary["outer_test_evaluated"] is False
```

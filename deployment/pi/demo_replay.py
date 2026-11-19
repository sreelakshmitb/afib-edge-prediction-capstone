from pathlib import Path
import hashlib
import json
import sys

import numpy as np
import onnxruntime as ort
from scipy.special import expit
from wfdb.processing import xqrs_detect

ROOT = Path(__file__).resolve().parent
sys.path.insert(
    0,
    str(ROOT / "_training_src/model_bundles/se_last_mask_seed7/code/src"),
)
from afib_edge.signal import bandpass_filter
from afib_edge.hrv import compute_hrv_features
from afib_edge.hrv_preprocess import HRVFoldPreprocessor

FILES = {
    "POS102": (
        "05494a672cc8fada857006410dc036f9235dd042caaf61256e0e1bb48131db71"
    ),
    "NEG024": (
        "e7d2360f128ddf5fbda2c00eb3633f5ebe246ba63ee12227de78fb290ca1db6d"
    ),
}
FS = 250
SEGMENT = 7500
REFERENCE_POS_SCORE = 0.3562

state = json.loads((ROOT / "model/hrv_preprocessor.json").read_text())
preprocessor = HRVFoldPreprocessor.from_state_dict(state)
session = ort.InferenceSession(
    str(ROOT / "model/forecast_model.onnx"),
    providers=["CPUExecutionProvider"],
)
inputs = session.get_inputs()
assert len(inputs) == 2, "Expected ECG and HRV inputs"
print("Model inputs:", [(item.name, item.shape) for item in inputs], flush=True)

def score(samples_uv, apply_filter):
    # WFDB training ECG is in mV; the replay files hold integer microvolts.
    ecg = samples_uv.astype(np.float64) / 1000.0
    if apply_filter:
        ecg = bandpass_filter(ecg, FS)
    else:
        ecg = ecg.astype(np.float32)

    peaks = np.asarray(
        xqrs_detect(sig=ecg.astype(np.float64), fs=FS, verbose=False),
        dtype=np.int64,
    )
    print(f"  R-peaks: {len(peaks)}", flush=True)

    segments = ecg.reshape(20, SEGMENT).copy()
    hrv = np.empty((20, 6), dtype=np.float32)
    for i in range(20):
        lo, hi = i * SEGMENT, (i + 1) * SEGMENT
        local = peaks[(peaks >= lo) & (peaks < hi)] - lo
        hrv[i] = compute_hrv_features(local, FS).values

    mean = segments.mean(axis=-1, keepdims=True)
    std = segments.std(axis=-1, keepdims=True)
    segments = (segments - mean) / np.maximum(std, 1e-6)
    ecg_input = segments.reshape(1, 20, 1, SEGMENT).astype(np.float32)
    hrv_input = preprocessor.transform(hrv[None, :, :])

    logit = float(
        np.asarray(
            session.run(
                None,
                {
                    inputs[0].name: ecg_input,
                    inputs[1].name: hrv_input,
                },
            )[0]
        ).reshape(-1)[0]
    )
    return float(expit(logit))

if len(sys.argv) != 2 or sys.argv[1].upper() not in FILES:
    raise SystemExit("Use: python demo_replay.py POS102  OR  python demo_replay.py NEG024")
choice = sys.argv[1].upper()
for name, expected_hash in [(choice, FILES[choice])]:
    path = ROOT / "captures" / f"{name}.BIN"
    data = path.read_bytes()
    actual_hash = hashlib.sha256(data).hexdigest()
    if len(data) != 600000 or actual_hash != expected_hash:
        raise RuntimeError(f"{name}: size or SHA-256 mismatch")
    samples = np.frombuffer(data, dtype="<i4")
    assert samples.shape == (150000,)
    print(f"\n{name}: capture checksum PASS", flush=True)

    for label, apply_filter in (("training filter", True),):
        print(f"{label}:", flush=True)
        probability = score(samples, apply_filter)
        message = f"  probability = {probability:.8f}"
        if name == "POS102":
            message += (
                f"  | difference from recorded ~0.3562 = "
                f"{abs(probability - REFERENCE_POS_SCORE):.8f}"
            )
        print(message, flush=True)
        threshold = 0.3310546875
        decision = "ELEVATED FORECAST RISK" if probability >= threshold else "BELOW FORECAST THRESHOLD"
        print(f"  {decision} (threshold {threshold:.8f})", flush=True)

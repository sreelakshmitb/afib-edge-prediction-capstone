"""Execute export fixtures; optionally reconstruct the exact trusted checkpoint."""
from pathlib import Path
import argparse
import hashlib
import json
import sys

import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT/'deployment/pi/model'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', action='store_true')
    args = parser.parse_args()
    metadata = json.loads((MODEL/'deployment_metadata.json').read_text())
    assert hashlib.sha256((MODEL/'forecast_model.onnx').read_bytes()).hexdigest() == metadata['onnx_sha256']
    session = ort.InferenceSession(str(MODEL/'forecast_model.onnx'), providers=['CPUExecutionProvider'])
    model = None
    if args.checkpoint:
        import torch
        torch.set_num_threads(2)
        sys.path.insert(0, str(ROOT/'training/unified/src'))
        from afib_unified.models import build_model
        checkpoint = ROOT/'artifacts/frozen_forecast_v1/forecast_checkpoint.pt'
        assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == metadata['source_checkpoint_sha256']
        payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
        model = build_model('proposed-groupnorm').eval()
        model.load_state_dict(payload['model_state_dict'], strict=True)
        assert sum(p.numel() for p in model.parameters()) == 159503
        assert not any(isinstance(m,torch.nn.BatchNorm1d) for m in model.modules())
    results=[]
    for expected in json.loads((MODEL/'validation_sample_expected.json').read_text()):
        with np.load(MODEL/expected['file'], allow_pickle=False) as sample:
            ecg = sample['ecg'].astype(np.float32)
            hrv = sample['hrv'].astype(np.float32)
        assert ecg.shape == (1,20,1,7500) and hrv.shape == (1,20,6)
        output = session.run(None, {session.get_inputs()[0].name: ecg, session.get_inputs()[1].name: hrv})[0]
        logit = float(np.asarray(output).reshape(-1)[0])
        probability = 1/(1+np.exp(-logit))
        assert abs(probability-expected['onnx_cpu_probability']) < 2e-5
        row = dict(file=expected['file'],onnx_probability=probability)
        if model is not None:
            with torch.inference_mode():
                pt = float(torch.sigmoid(model(torch.from_numpy(ecg),torch.from_numpy(hrv))).item())
            assert abs(pt-probability) < 2e-5
            row.update(pytorch_probability=pt, absolute_difference=abs(pt-probability))
        results.append(row)
    print(json.dumps({'status':'PASS','checkpoint_verified':args.checkpoint,'samples':results},indent=2))

if __name__ == '__main__':
    main()

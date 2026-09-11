"""FP32 export of a preselected trained fold; embeds HRV preprocessing."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch import nn

from afib.data import load_arrays
from afib.models import AFibModel


class InferenceModel(nn.Module):
    def __init__(self, model, transform):
        super().__init__()
        self.model = model
        for key in ('median', 'mean', 'scale'):
            self.register_buffer(key, torch.tensor(transform[key], dtype=torch.float32))

    def forward(self, ecg, hrv):
        hrv = torch.where(torch.isfinite(hrv), hrv, self.median)
        hrv = torch.clamp((hrv-self.mean)/self.scale, -10., 10.)
        return torch.sigmoid(self.model(ecg, hrv))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    args = parser.parse_args()
    path = Path(args.checkpoint).resolve()
    if Path(__file__).resolve().parents[1] in path.parents:
        raise ValueError('Checkpoint/export must be outside Git checkout')
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if checkpoint['config']['max_steps']:
        raise ValueError('Refusing to export a smoke-only checkpoint as deployment candidate')
    torch.set_num_threads(2)
    model = AFibModel(se=checkpoint['config']['model']=='se')
    model.load_state_dict(checkpoint['model'])
    wrapper = InferenceModel(model, checkpoint['transform']).eval()
    manifest_path = Path(checkpoint['config']['root']) / 'manifest.json'
    if hashlib.sha256(manifest_path.read_bytes()).hexdigest() != checkpoint['config']['manifest_sha256']:
        raise ValueError('Manifest changed since training; cannot resolve validation examples safely')
    manifest = json.loads(manifest_path.read_text())
    split = json.loads((path.parent / 'split.json').read_text())
    # Parity uses inner-validation data, not outer-test performance for selection.
    examples = [load_arrays(manifest[i]) for i in split['validation']['indices'][:2]]
    ecg = torch.from_numpy(np.stack([x[0] for x in examples]))
    hrv = torch.from_numpy(np.stack([x[1] for x in examples]))
    output = path.with_suffix('.onnx')
    if output.exists():
        raise FileExistsError(output)
    torch.onnx.export(wrapper, (ecg[:1], hrv[:1]), str(output), input_names=['ecg', 'hrv'],
        output_names=['af_probability'], dynamic_axes={'ecg': {0:'batch'}, 'hrv':{0:'batch'},
        'af_probability':{0:'batch'}}, opset_version=17, dynamo=False)
    onnx.checker.check_model(onnx.load(str(output)))
    session = ort.InferenceSession(str(output), providers=['CPUExecutionProvider'])
    errors = []
    for batch in range(1, len(examples)+1):
        with torch.no_grad():
            expected = wrapper(ecg[:batch], hrv[:batch]).numpy()
        actual = session.run(None, {'ecg':ecg[:batch].numpy(), 'hrv':hrv[:batch].numpy()})[0]
        np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)
        errors.append(float(np.max(np.abs(actual-expected))))
    report = dict(checkpoint=str(path), output=str(output), bytes=output.stat().st_size,
        parameters=sum(p.numel() for p in wrapper.parameters()), batches_tested=list(range(1,len(examples)+1)),
        max_absolute_error=max(errors), dtype='float32', threshold=.5,
        input_contract='Preprocessed normalized ECG [B,20,1,7500]; raw HRV with NaNs [B,20,6]',
        hrv_order=checkpoint['config']['hrv_order'], hrv_transform_embedded=True,
        selection=checkpoint['selection'],
        raspberry_pi_latency_ms=None)
    output.with_suffix('.export.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

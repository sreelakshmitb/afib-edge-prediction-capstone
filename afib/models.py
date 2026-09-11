"""Small reference-family CNN-HRV-UniLSTM and primary SE variant.

Three convolutions, 128-d embedding and a 128-state UniLSTM follow the paper.
Undisclosed channel widths/strides are documented engineering choices.
"""
import torch
from torch import nn


class SqueezeExcitation(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(channels, max(4, channels//4)), nn.ReLU(),
                                  nn.Linear(max(4, channels//4), channels), nn.Sigmoid())

    def forward(self, x):
        return x * self.gate(x.mean(dim=-1)).unsqueeze(-1)


class AFibModel(nn.Module):
    def __init__(self, se=True):
        super().__init__()
        blocks = []
        for incoming, outgoing, kernel, stride in [(1, 32, 7, 5), (32, 64, 5, 5), (64, 128, 5, 5)]:
            blocks.extend([nn.Conv1d(incoming, outgoing, kernel, stride=stride, padding=kernel//2),
                           nn.BatchNorm1d(outgoing), nn.ReLU()])
            if se:
                blocks.append(SqueezeExcitation(outgoing))
        self.encoder = nn.Sequential(*blocks, nn.AdaptiveAvgPool1d(1))
        self.lstm = nn.LSTM(128 + 6, 128, batch_first=True, bidirectional=False)
        self.head = nn.Sequential(nn.Dropout(.3), nn.Linear(128, 1))
        if sum(p.numel() for p in self.parameters()) >= 250000:
            raise ValueError('Parameter budget exceeded')

    def forward(self, ecg, hrv):
        if not torch.jit.is_tracing():
            if ecg.ndim != 4 or tuple(ecg.shape[1:]) != (20, 1, 7500):
                raise ValueError('ECG contract is [B,20,1,7500]')
            if tuple(hrv.shape) != (ecg.shape[0], 20, 6):
                raise ValueError('HRV contract is [B,20,6]')
        batch = ecg.shape[0]
        encoded = self.encoder(ecg.reshape(-1, 1, 7500)).reshape(batch, 20, 128)
        sequence, _ = self.lstm(torch.cat([encoded, hrv], dim=-1))
        return self.head(sequence.mean(dim=1)).squeeze(-1)

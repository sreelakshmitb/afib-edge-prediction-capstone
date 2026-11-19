from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class DetectorConfig:
    embedding_dim: int = 128
    dropout: float = 0.30
    channels: tuple[int, ...] = (16, 32, 64)
    kernels: tuple[int, ...] = (7, 7, 5)
    se_reduction: int = 8


class SE1D(nn.Module):
    def __init__(self, channels: int, reduction: int) -> None:
        super().__init__()
        reduced = max(1, channels // reduction)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.gate = nn.Sequential(
            nn.Linear(channels, reduced), nn.ReLU(inplace=True),
            nn.Linear(reduced, channels), nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gate(self.pool(x).flatten(1)).unsqueeze(-1)


class CurrentAFDetector(nn.Module):
    """Small 30-second normal-vs-current-AF detector."""

    def __init__(self, config: DetectorConfig | None = None) -> None:
        super().__init__()
        self.config = config or DetectorConfig()
        layers: list[nn.Module] = []
        incoming = 1
        for outgoing, kernel in zip(self.config.channels, self.config.kernels):
            layers.extend([
                nn.Conv1d(incoming, outgoing, kernel, padding=kernel // 2, bias=False),
                nn.BatchNorm1d(outgoing),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(2),
                SE1D(outgoing, self.config.se_reduction),
            ])
            incoming = outgoing
        self.features = nn.Sequential(*layers)
        self.projection = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), nn.Flatten(1),
            nn.Linear(self.config.channels[-1], self.config.embedding_dim),
            nn.LayerNorm(self.config.embedding_dim), nn.ReLU(inplace=True),
        )
        self.dropout = nn.Dropout(self.config.dropout)
        self.classifier = nn.Linear(self.config.embedding_dim, 1)

    def forward(self, ecg: torch.Tensor) -> torch.Tensor:
        if ecg.ndim != 3 or tuple(ecg.shape[1:]) != (1, 7500):
            raise ValueError("ECG must have shape [B,1,7500]")
        return self.classifier(self.dropout(self.projection(self.features(ecg)))).squeeze(-1)

    @property
    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def checkpoint_config(self) -> dict:
        return asdict(self.config)

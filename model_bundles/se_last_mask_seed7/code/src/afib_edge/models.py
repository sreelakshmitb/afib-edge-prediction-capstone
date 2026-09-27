"""Paper-aligned lightweight CNN-HRV-UniLSTM models.

The model consumes one ten-minute observation as twenty 30-second segments.
Each ECG segment is encoded independently, concatenated with its six HRV
features, and aggregated causally by a single unidirectional LSTM.

HRV imputation and feature scaling deliberately live outside this module. They
must be fitted on the training patients of each outer fold only.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .contracts import DEFAULT_CONTRACT, HRV_FEATURE_NAMES


MAX_TRAINABLE_PARAMETERS = 250_000


@dataclass(frozen=True)
class ModelConfig:
    """Architecture settings with the project defaults fixed by the paper."""

    embedding_dim: int = 128
    hidden_dim: int = 128
    dropout: float = 0.30
    channels: tuple[int, ...] = (16, 32, 64)
    kernels: tuple[int, ...] = (7, 7, 5)
    se_reduction: int = 8

    def validate(self) -> None:
        if self.embedding_dim <= 0 or self.hidden_dim <= 0:
            raise ValueError("embedding_dim and hidden_dim must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if len(self.channels) != 3 or len(self.kernels) != 3:
            raise ValueError("the fixed encoder has exactly three convolutional layers")
        if any(channel <= 0 for channel in self.channels):
            raise ValueError("all convolutional channel counts must be positive")
        if any(kernel <= 0 or kernel % 2 == 0 for kernel in self.kernels):
            raise ValueError("convolution kernels must be positive odd integers")
        if self.se_reduction <= 0:
            raise ValueError("se_reduction must be positive")


class SqueezeExcitation1D(nn.Module):
    """Channel recalibration for a one-dimensional convolutional feature map."""

    def __init__(self, channels: int, reduction: int) -> None:
        super().__init__()
        reduced = max(1, channels // reduction)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.gate = nn.Sequential(
            nn.Linear(channels, reduced),
            nn.ReLU(inplace=True),
            nn.Linear(reduced, channels),
            nn.Sigmoid(),
        )

    def forward(self, features: Tensor) -> Tensor:
        weights = self.gate(self.pool(features).flatten(1)).unsqueeze(-1)
        return features * weights


class ECGSegmentEncoder(nn.Module):
    """Three-layer 1-D CNN producing one 128-D morphology vector per segment."""

    def __init__(self, config: ModelConfig, use_se: bool) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_channels = 1
        for out_channels, kernel in zip(config.channels, config.kernels):
            layers.extend(
                [
                    nn.Conv1d(in_channels, out_channels, kernel, padding=kernel // 2, bias=False),
                    nn.BatchNorm1d(out_channels),
                    nn.ReLU(inplace=True),
                    nn.MaxPool1d(kernel_size=2),
                ]
            )
            if use_se:
                layers.append(SqueezeExcitation1D(out_channels, config.se_reduction))
            in_channels = out_channels

        self.features = nn.Sequential(*layers)
        self.projection = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(start_dim=1),
            nn.Linear(config.channels[-1], config.embedding_dim),
            nn.LayerNorm(config.embedding_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, segment: Tensor) -> Tensor:
        return self.projection(self.features(segment))


class CNNHRVUniLSTM(nn.Module):
    """Hybrid causal sequence model with an optional squeeze-excitation CNN."""

    def __init__(self, config: ModelConfig | None = None, use_se: bool = False) -> None:
        super().__init__()
        self.config = config or ModelConfig()
        self.config.validate()
        self.use_se = use_se

        self.ecg_encoder = ECGSegmentEncoder(self.config, use_se=use_se)
        self.temporal = nn.LSTM(
            input_size=self.config.embedding_dim + len(HRV_FEATURE_NAMES),
            hidden_size=self.config.hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
        )
        self.input_dropout = nn.Dropout(self.config.dropout)
        self.output_dropout = nn.Dropout(self.config.dropout)
        self.classifier = nn.Linear(self.config.hidden_dim, 1)

    @property
    def parameter_count(self) -> int:
        """Number of trainable parameters used for the deployment budget."""
        return count_parameters(self)

    def _validate_inputs(self, ecg: Tensor, hrv: Tensor) -> None:
        expected_ecg = (
            DEFAULT_CONTRACT.segments_per_window,
            1,
            DEFAULT_CONTRACT.segment_samples,
        )
        expected_hrv = (DEFAULT_CONTRACT.segments_per_window, len(HRV_FEATURE_NAMES))
        if ecg.ndim != 4 or tuple(ecg.shape[1:]) != expected_ecg:
            raise ValueError(f"ECG must have shape [B,{expected_ecg[0]},1,{expected_ecg[2]}]")
        if hrv.ndim != 3 or tuple(hrv.shape[1:]) != expected_hrv:
            raise ValueError(f"HRV must have shape [B,{expected_hrv[0]},{expected_hrv[1]}]")
        if ecg.shape[0] != hrv.shape[0]:
            raise ValueError("ECG and HRV batch dimensions must match")
        if not torch.is_floating_point(ecg) or not torch.is_floating_point(hrv):
            raise TypeError("ECG and HRV tensors must be floating point")
        if not torch.isfinite(hrv).all():
            raise ValueError("HRV contains non-finite values; impute using training-fold statistics")

    def forward(self, ecg: Tensor, hrv: Tensor) -> Tensor:
        """Return one uncalibrated AF logit per observation window."""
        self._validate_inputs(ecg, hrv)
        batch_size, steps = ecg.shape[:2]

        segments = ecg.reshape(batch_size * steps, 1, DEFAULT_CONTRACT.segment_samples)
        morphology = self.ecg_encoder(segments).reshape(batch_size, steps, -1)
        fused = torch.cat((morphology, hrv), dim=-1)
        sequence, _ = self.temporal(self.input_dropout(fused))
        representation = self.output_dropout(sequence[:, -1, :])
        return self.classifier(representation).squeeze(-1)

    def predict_proba(self, ecg: Tensor, hrv: Tensor) -> Tensor:
        """Return sigmoid probabilities; calibration is applied outside the model."""
        return torch.sigmoid(self(ecg, hrv))


class PaperCNNHRVUniLSTM(CNNHRVUniLSTM):
    """Reference model: CNN morphology + HRV + one UniLSTM."""

    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__(config=config, use_se=False)


class SECNNHRVUniLSTM(CNNHRVUniLSTM):
    """Proposed model: the reference encoder augmented with SE channel gates."""

    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__(config=config, use_se=True)


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """Count parameters, optionally restricted to parameters requiring gradients."""
    parameters = model.parameters()
    if trainable_only:
        parameters = (parameter for parameter in parameters if parameter.requires_grad)
    return sum(parameter.numel() for parameter in parameters)


def build_model(name: str, config: ModelConfig | None = None) -> CNNHRVUniLSTM:
    """Build a named model for config files and training scripts."""
    normalized = name.strip().lower().replace("_", "-")
    if normalized in {"paper", "cnn-hrv-unilstm", "reference"}:
        model = PaperCNNHRVUniLSTM(config=config)
    elif normalized in {"se", "se-cnn-hrv-unilstm", "proposed"}:
        model = SECNNHRVUniLSTM(config=config)
    else:
        raise ValueError(f"unknown model name: {name}")
    if model.parameter_count >= MAX_TRAINABLE_PARAMETERS:
        raise ValueError(
            f"{name} has {model.parameter_count:,} trainable parameters; "
            f"the limit is {MAX_TRAINABLE_PARAMETERS:,}"
        )
    return model


__all__ = [
    "CNNHRVUniLSTM",
    "ECGSegmentEncoder",
    "MAX_TRAINABLE_PARAMETERS",
    "ModelConfig",
    "PaperCNNHRVUniLSTM",
    "SECNNHRVUniLSTM",
    "SqueezeExcitation1D",
    "build_model",
    "count_parameters",
]

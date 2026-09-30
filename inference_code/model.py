from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, Tuple

import torch
from torch import nn
from torch.nn import functional as F


def parse_int_list(
    value: str | Sequence[int],
) -> tuple[int, ...]:
    if isinstance(value, str):
        result = tuple(
            int(item.strip())
            for item in value.split(",")
            if item.strip()
        )
    else:
        result = tuple(int(item) for item in value)

    if not result:
        raise ValueError("The kernel list cannot be empty.")
    if any(item <= 0 for item in result):
        raise ValueError("All kernel sizes must be positive.")
    if any(item % 2 == 0 for item in result):
        raise ValueError(
            "Temporal kernel sizes must be odd so time length is preserved."
        )
    return result


def length_padding_mask(
    lengths: torch.Tensor,
    max_length: int,
) -> torch.Tensor:
    positions = torch.arange(
        max_length,
        device=lengths.device,
    ).unsqueeze(0)
    return positions >= lengths.unsqueeze(1)


class SinusoidalPositionEncoding(nn.Module):
    def __init__(
        self,
        d_model: int,
        max_length: int = 2048,
    ) -> None:
        super().__init__()

        position = torch.arange(
            max_length,
            dtype=torch.float32,
        ).unsqueeze(1)
        scale = torch.exp(
            torch.arange(
                0,
                d_model,
                2,
                dtype=torch.float32,
            )
            * (-math.log(10000.0) / d_model)
        )

        encoding = torch.zeros(
            max_length,
            d_model,
            dtype=torch.float32,
        )
        encoding[:, 0::2] = torch.sin(position * scale)

        if d_model > 1:
            encoding[:, 1::2] = torch.cos(
                position
                * scale[: encoding[:, 1::2].shape[1]]
            )

        self.register_buffer(
            "encoding",
            encoding,
            persistent=False,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        if x.shape[1] > self.encoding.shape[0]:
            raise ValueError(
                f"Sequence length {x.shape[1]} exceeds "
                f"max_length={self.encoding.shape[0]}."
            )

        return (
            x
            + self.encoding[: x.shape[1]]
            .unsqueeze(0)
            .to(dtype=x.dtype)
        )


class FullBandSpectralBranch(nn.Module):
    """2-D frequency filter that preserves every time step.

    Input:
        [B, 1, T, F]

    The convolution stride is always (1, 1). The temporal dimension is
    padded so an odd temporal kernel keeps T unchanged. The frequency
    kernel covers the full spectrogram band by default. If a frequency
    kernel larger than F is requested, symmetric zero padding is applied
    before convolution. Any residual frequency positions are averaged;
    no time pooling is performed.
    """

    def __init__(
        self,
        out_channels: int,
        temporal_kernel: int,
        frequency_kernel: int,
        dropout: float,
    ) -> None:
        super().__init__()

        if temporal_kernel <= 0 or temporal_kernel % 2 == 0:
            raise ValueError(
                "temporal_kernel must be a positive odd integer."
            )
        if frequency_kernel <= 0:
            raise ValueError(
                "frequency_kernel must be positive."
            )

        self.temporal_kernel = int(temporal_kernel)
        self.frequency_kernel = int(frequency_kernel)

        self.conv = nn.Conv2d(
            in_channels=1,
            out_channels=out_channels,
            kernel_size=(
                self.temporal_kernel,
                self.frequency_kernel,
            ),
            stride=(1, 1),
            padding=(self.temporal_kernel // 2, 0),
            bias=False,
        )
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout2d(dropout)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        frequency_bins = x.shape[-1]

        if self.frequency_kernel > frequency_bins:
            total_pad = self.frequency_kernel - frequency_bins
            pad_left = total_pad // 2
            pad_right = total_pad - pad_left
            x = F.pad(
                x,
                (pad_left, pad_right, 0, 0),
            )

        if self.frequency_kernel < x.shape[-1]:
            # The branch remains valid for narrower kernels. Frequency is
            # collapsed after convolution, while time is left untouched.
            y = self.conv(x)
            y = y.mean(dim=-1, keepdim=True)
        else:
            y = self.conv(x)

        y = self.dropout(
            self.activation(
                self.norm(y)
            )
        )

        # [B,C,T,1] -> [B,T,C]
        return y.squeeze(-1).transpose(1, 2)


def make_transformer_encoder(
    d_model: int,
    nhead: int,
    num_layers: int,
    ffn_dim: int,
    dropout: float,
) -> nn.TransformerEncoder:
    if d_model % nhead != 0:
        raise ValueError(
            "d_model must be divisible by nhead."
        )

    encoder_layer = nn.TransformerEncoderLayer(
        d_model=d_model,
        nhead=nhead,
        dim_feedforward=ffn_dim,
        dropout=dropout,
        activation="gelu",
        batch_first=True,
        norm_first=True,
    )

    kwargs = {
        "encoder_layer": encoder_layer,
        "num_layers": num_layers,
        "norm": nn.LayerNorm(d_model),
    }

    try:
        return nn.TransformerEncoder(
            **kwargs,
            enable_nested_tensor=False,
        )
    except TypeError:
        return nn.TransformerEncoder(**kwargs)


class JointSpectralTransformer(nn.Module):
    """Frequency CNN + temporal Transformer for joint I/E prediction.

    Processing
    ----------
    1. Split the input into:
       - 129-bin standardized log-power spectrogram
       - 64 auxiliary features (MFCC, deltas, band energies)
    2. Apply full-band 2-D spectral filters with stride=(1,1).
       Default filters are (1,129), (3,129), and (5,129).
       Every branch keeps all 938 time positions.
    3. Project auxiliary features independently at each time position.
    4. Fuse spectral and auxiliary features.
    5. Apply a Transformer along the complete time sequence.
    6. Produce I and E logits at every original frame.

    Input:
        [B, T=938, F=193]

    Output:
        [B, T=938, 2]
    """

    def __init__(
        self,
        feature_dim: int = 193,
        spectrogram_bins: int = 129,
        spectral_time_kernels: Sequence[int] = (1, 3, 5),
        spectral_frequency_kernel: int = 129,
        spectral_branch_channels: int = 64,
        auxiliary_channels: int = 64,
        d_model: int = 256,
        nhead: int = 8,
        transformer_layers: int = 3,
        ffn_dim: int = 1024,
        dropout: float = 0.2,
        max_length: int = 2048,
    ) -> None:
        super().__init__()

        if spectrogram_bins <= 0:
            raise ValueError("spectrogram_bins must be positive.")
        if feature_dim <= spectrogram_bins:
            raise ValueError(
                "feature_dim must include spectrogram and auxiliary features."
            )

        temporal_kernels = parse_int_list(
            spectral_time_kernels
        )

        self.feature_dim = int(feature_dim)
        self.spectrogram_bins = int(spectrogram_bins)
        self.auxiliary_dim = (
            self.feature_dim - self.spectrogram_bins
        )

        self.spectral_branches = nn.ModuleList(
            [
                FullBandSpectralBranch(
                    out_channels=spectral_branch_channels,
                    temporal_kernel=kernel,
                    frequency_kernel=spectral_frequency_kernel,
                    dropout=dropout,
                )
                for kernel in temporal_kernels
            ]
        )

        spectral_output_dim = (
            spectral_branch_channels
            * len(self.spectral_branches)
        )

        self.auxiliary_projection = nn.Sequential(
            nn.LayerNorm(self.auxiliary_dim),
            nn.Linear(
                self.auxiliary_dim,
                auxiliary_channels,
            ),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        fused_dim = (
            spectral_output_dim
            + auxiliary_channels
        )

        self.fusion = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Linear(fused_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.position = SinusoidalPositionEncoding(
            d_model=d_model,
            max_length=max_length,
        )
        self.transformer = make_transformer_encoder(
            d_model=d_model,
            nhead=nhead,
            num_layers=transformer_layers,
            ffn_dim=ffn_dim,
            dropout=dropout,
        )
        self.output_head = nn.Linear(
            d_model,
            2,
        )

    def forward(
        self,
        x: torch.Tensor,
        input_lengths: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 3:
            raise ValueError(
                f"Expected [B,T,F], got {tuple(x.shape)}"
            )
        if x.shape[-1] != self.feature_dim:
            raise ValueError(
                f"Expected feature_dim={self.feature_dim}, "
                f"got {x.shape[-1]}."
            )

        batch_size, time_steps, _ = x.shape

        if input_lengths is None:
            input_lengths = torch.full(
                (batch_size,),
                time_steps,
                dtype=torch.long,
                device=x.device,
            )
        else:
            input_lengths = input_lengths.to(
                device=x.device,
                dtype=torch.long,
            )

        padding_mask = length_padding_mask(
            input_lengths,
            time_steps,
        )

        spectrogram = x[
            :,
            :,
            : self.spectrogram_bins,
        ].unsqueeze(1)

        auxiliary = x[
            :,
            :,
            self.spectrogram_bins :,
        ]

        spectral_features = torch.cat(
            [
                branch(spectrogram)
                for branch in self.spectral_branches
            ],
            dim=-1,
        )

        auxiliary_features = (
            self.auxiliary_projection(
                auxiliary
            )
        )

        features = torch.cat(
            [
                spectral_features,
                auxiliary_features,
            ],
            dim=-1,
        )
        features = self.fusion(features)
        features = features.masked_fill(
            padding_mask.unsqueeze(-1),
            0.0,
        )

        features = self.position(features)
        features = self.transformer(
            features,
            src_key_padding_mask=padding_mask,
        )
        logits = self.output_head(features)
        logits = logits.masked_fill(
            padding_mask.unsqueeze(-1),
            0.0,
        )

        return logits, input_lengths


def build_model(
    model_name: str,
    feature_dim: int,
    spectrogram_bins: int = 129,
    spectral_time_kernels: str | Sequence[int] = "1,3,5",
    spectral_frequency_kernel: int = 129,
    spectral_branch_channels: int = 64,
    auxiliary_channels: int = 64,
    d_model: int = 256,
    nhead: int = 8,
    transformer_layers: int = 3,
    ffn_dim: int = 1024,
    dropout: float = 0.2,
    max_length: int = 2048,
    **_: Any,
) -> nn.Module:
    name = model_name.lower()

    if name != "spectral_transformer":
        raise ValueError(
            "This project implements only 'spectral_transformer'."
        )

    return JointSpectralTransformer(
        feature_dim=feature_dim,
        spectrogram_bins=spectrogram_bins,
        spectral_time_kernels=parse_int_list(
            spectral_time_kernels
        ),
        spectral_frequency_kernel=(
            spectral_frequency_kernel
        ),
        spectral_branch_channels=(
            spectral_branch_channels
        ),
        auxiliary_channels=auxiliary_channels,
        d_model=d_model,
        nhead=nhead,
        transformer_layers=transformer_layers,
        ffn_dim=ffn_dim,
        dropout=dropout,
        max_length=max_length,
    )


def build_model_from_config(
    config: Mapping[str, Any],
    feature_dim: int,
) -> nn.Module:
    return build_model(
        model_name=str(
            config.get(
                "model",
                "spectral_transformer",
            )
        ),
        feature_dim=feature_dim,
        spectrogram_bins=int(
            config.get(
                "spectrogram_bins",
                129,
            )
        ),
        spectral_time_kernels=config.get(
            "spectral_time_kernels",
            "1,3,5",
        ),
        spectral_frequency_kernel=int(
            config.get(
                "spectral_frequency_kernel",
                129,
            )
        ),
        spectral_branch_channels=int(
            config.get(
                "spectral_branch_channels",
                64,
            )
        ),
        auxiliary_channels=int(
            config.get(
                "auxiliary_channels",
                64,
            )
        ),
        d_model=int(
            config.get(
                "d_model",
                256,
            )
        ),
        nhead=int(
            config.get(
                "nhead",
                8,
            )
        ),
        transformer_layers=int(
            config.get(
                "transformer_layers",
                3,
            )
        ),
        ffn_dim=int(
            config.get(
                "ffn_dim",
                1024,
            )
        ),
        dropout=float(
            config.get(
                "dropout",
                0.2,
            )
        ),
        max_length=int(
            config.get(
                "max_length",
                2048,
            )
        ),
    )

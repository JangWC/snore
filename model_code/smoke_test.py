from __future__ import annotations

import torch

from loss import MaskedJointBCELoss
from model import build_model


def main() -> None:
    model = build_model(
        model_name="spectral_transformer",
        feature_dim=193,
        spectrogram_bins=129,
        spectral_time_kernels="1,3,5",
        spectral_frequency_kernel=129,
        spectral_branch_channels=8,
        auxiliary_channels=8,
        d_model=32,
        nhead=4,
        transformer_layers=1,
        ffn_dim=64,
        dropout=0.1,
    )

    features = torch.randn(
        2,
        938,
        193,
    )
    targets = torch.randint(
        0,
        2,
        (2, 938, 2),
    ).float()
    lengths = torch.tensor(
        [938, 931],
        dtype=torch.long,
    )

    logits, output_lengths = model(
        features,
        lengths,
    )
    loss = MaskedJointBCELoss()(
        logits,
        targets,
        output_lengths,
    )

    assert logits.shape == (
        2,
        938,
        2,
    ), logits.shape
    assert output_lengths.tolist() == [
        938,
        931,
    ]
    assert torch.isfinite(loss)

    parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print("logits:", tuple(logits.shape))
    print(
        "output lengths:",
        output_lengths.tolist(),
    )
    print("parameters:", f"{parameters:,}")
    print("loss:", float(loss.item()))


if __name__ == "__main__":
    main()

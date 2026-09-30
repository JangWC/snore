from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from build_split import build_split
from dataset import compute_pos_weight, datasets_from_split
from loss import MaskedJointBCELoss
from metrics import choose_threshold, evaluate_joint
from model import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a full-resolution frequency-CNN + temporal-Transformer "
            "joint I/E model."
        )
    )

    parser.add_argument("--train-h5", required=True)
    parser.add_argument("--test-h5", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split-file", default=None)
    parser.add_argument("--rebuild-split", action="store_true")
    parser.add_argument("--input-key", default="features")
    parser.add_argument(
        "--label-key",
        default="frame_target",
        help=(
            "Use frame_target to preserve all 938 time positions. "
            "Do not use target for this model."
        ),
    )
    parser.add_argument("--group-key", default="subject_id")
    parser.add_argument("--val-ratio", type=float, default=0.15)

    parser.add_argument(
        "--model",
        choices=["spectral_transformer"],
        default="spectral_transformer",
    )
    parser.add_argument("--spectrogram-bins", type=int, default=129)
    parser.add_argument(
        "--spectral-time-kernels",
        default="1,3,5",
        help=(
            "Comma-separated odd temporal widths. Each branch uses a "
            "full-band frequency filter and stride=(1,1)."
        ),
    )
    parser.add_argument(
        "--spectral-frequency-kernel",
        type=int,
        default=129,
    )
    parser.add_argument(
        "--spectral-branch-channels",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--auxiliary-channels",
        type=int,
        default=64,
    )
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument(
        "--transformer-layers",
        type=int,
        default=3,
    )
    parser.add_argument("--ffn-dim", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--max-length", type=int, default=2048)

    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum-steps", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--lr-patience", type=int, default=10)
    parser.add_argument("--lr-factor", type=float, default=0.2)
    parser.add_argument("--early-stopping", type=int, default=50)
    parser.add_argument(
        "--monitor",
        choices=["mean_accuracy", "mean_f1", "val_loss"],
        default="mean_f1",
    )
    parser.add_argument(
        "--threshold-metric",
        choices=["accuracy", "f1"],
        default="f1",
    )
    parser.add_argument(
        "--pos-weight",
        choices=["none", "balanced"],
        default="none",
    )

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp", action="store_true")
    return parser.parse_args()


def choose_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
    return torch.device(value)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_grad_scaler(
    enabled: bool,
):
    try:
        return torch.amp.GradScaler(
            "cuda",
            enabled=enabled,
        )
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(
            enabled=enabled,
        )


def save_json(
    value: Any,
    path: Path,
) -> None:
    def convert(obj: Any):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(
            f"Cannot serialize {type(obj)}"
        )

    path.write_text(
        json.dumps(
            value,
            indent=2,
            ensure_ascii=False,
            default=convert,
        ),
        encoding="utf-8",
    )


def append_history(
    path: Path,
    row: dict[str, Any],
) -> None:
    write_header = not path.exists()

    with path.open(
        "a",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(row.keys()),
        )
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def flatten_valid(
    tensor: torch.Tensor,
    lengths: torch.Tensor,
) -> np.ndarray:
    parts = [
        tensor[
            index,
            : int(lengths[index]),
        ]
        .detach()
        .cpu()
        .numpy()
        for index in range(tensor.shape[0])
    ]
    return np.concatenate(parts, axis=0)


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    scaler,
    amp_enabled: bool,
    grad_accum_steps: int,
) -> tuple[float, np.ndarray, np.ndarray]:
    training = optimizer is not None
    model.train(training)

    if training:
        optimizer.zero_grad(set_to_none=True)

    total_loss = 0.0
    batch_count = 0
    targets: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []

    for step, batch in enumerate(loader, start=1):
        (
            features,
            target,
            input_lengths,
            _,
            _,
            _,
        ) = batch

        features = features.to(
            device,
            non_blocking=True,
        )
        target = target.to(
            device,
            non_blocking=True,
        )
        input_lengths = input_lengths.to(
            device,
            non_blocking=True,
        )

        with torch.set_grad_enabled(training):
            with torch.autocast(
                device_type=device.type,
                enabled=amp_enabled,
            ):
                logits, output_lengths = model(
                    features,
                    input_lengths,
                )

                if logits.shape != target.shape:
                    raise RuntimeError(
                        f"Model output {tuple(logits.shape)} "
                        f"does not match target {tuple(target.shape)}. "
                        "Use --label-key frame_target."
                    )

                loss = criterion(
                    logits,
                    target,
                    output_lengths,
                )
                backward_loss = (
                    loss / max(grad_accum_steps, 1)
                )

            if training:
                if amp_enabled:
                    scaler.scale(
                        backward_loss
                    ).backward()
                else:
                    backward_loss.backward()

                should_step = (
                    step % grad_accum_steps == 0
                    or step == len(loader)
                )

                if should_step:
                    if amp_enabled:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()

                    optimizer.zero_grad(
                        set_to_none=True
                    )

        total_loss += float(loss.item())
        batch_count += 1

        targets.append(
            flatten_valid(
                target,
                output_lengths,
            )
        )
        probabilities.append(
            flatten_valid(
                torch.sigmoid(logits),
                output_lengths,
            )
        )

    return (
        total_loss / max(batch_count, 1),
        np.concatenate(targets, axis=0),
        np.concatenate(probabilities, axis=0),
    )


def main() -> None:
    args = parse_args()

    if args.label_key != "frame_target":
        raise ValueError(
            "This model preserves 938 time steps and must use "
            "--label-key frame_target."
        )

    set_seed(args.seed)
    device = choose_device(args.device)
    amp_enabled = bool(
        args.amp and device.type == "cuda"
    )

    output_dir = Path(
        args.output_dir
    ).resolve()
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Every run stores its own local split file, even when reusing a
    # split from another experiment.
    local_split_file = output_dir / "split_indices.npz"
    source_split_file = (
        Path(args.split_file).resolve()
        if args.split_file
        else None
    )

    def create_local_split() -> None:
        summary = build_split(
            train_h5=args.train_h5,
            test_h5=args.test_h5,
            output_path=local_split_file,
            val_ratio=args.val_ratio,
            seed=args.seed,
            label_key=args.label_key,
            group_key=(
                args.group_key
                if args.group_key
                else None
            ),
        )
        save_json(
            summary,
            output_dir / "split_summary.json",
        )

    if args.rebuild_split:
        create_local_split()

    elif source_split_file is not None:
        if not source_split_file.exists():
            raise FileNotFoundError(
                f"Split file not found: {source_split_file}"
            )

        if source_split_file != local_split_file:
            shutil.copy2(
                source_split_file,
                local_split_file,
            )

        source_split_json = source_split_file.with_suffix(
            ".json"
        )
        local_split_json = local_split_file.with_suffix(
            ".json"
        )

        if source_split_json.exists():
            if source_split_json != local_split_json:
                shutil.copy2(
                    source_split_json,
                    local_split_json,
                )

            shutil.copy2(
                source_split_json,
                output_dir / "split_summary.json",
            )
        else:
            # Preserve a minimal record of which split was reused.
            save_json(
                {
                    "reused_split": str(source_split_file),
                    "saved_local_split": str(local_split_file),
                },
                output_dir / "split_summary.json",
            )

        print("Reused split:", source_split_file)
        print("Saved local split:", local_split_file)

    elif local_split_file.exists():
        print(
            "Using existing local split:",
            local_split_file,
        )

    else:
        create_local_split()

    # Training and checkpoints always refer to the split stored in this run.
    split_file = local_split_file

    train_set, val_set, _, metadata = (
        datasets_from_split(
            split_file,
            input_key=args.input_key,
            label_key=args.label_key,
        )
    )

    if metadata["input_frames"] != metadata["output_frames"]:
        raise RuntimeError(
            "This model requires equal input and label lengths. "
            f"Input={metadata['input_frames']}, "
            f"label={metadata['output_frames']}."
        )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(
            args.num_workers > 0
        ),
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(
            args.num_workers > 0
        ),
    )

    model = build_model(
        model_name=args.model,
        feature_dim=metadata["feature_dim"],
        spectrogram_bins=args.spectrogram_bins,
        spectral_time_kernels=(
            args.spectral_time_kernels
        ),
        spectral_frequency_kernel=(
            args.spectral_frequency_kernel
        ),
        spectral_branch_channels=(
            args.spectral_branch_channels
        ),
        auxiliary_channels=(
            args.auxiliary_channels
        ),
        d_model=args.d_model,
        nhead=args.nhead,
        transformer_layers=(
            args.transformer_layers
        ),
        ffn_dim=args.ffn_dim,
        dropout=args.dropout,
        max_length=args.max_length,
    ).to(device)

    pos_weight_tensor = None

    if args.pos_weight == "balanced":
        pos_weight, positive, negative = (
            compute_pos_weight(train_set)
        )
        pos_weight_tensor = torch.tensor(
            pos_weight,
            dtype=torch.float32,
            device=device,
        )
        print(
            "Balanced pos_weight | "
            f"I={pos_weight[0]:.4f}, "
            f"E={pos_weight[1]:.4f} | "
            f"positive={positive.tolist()} "
            f"negative={negative.tolist()}"
        )

    criterion = MaskedJointBCELoss(
        pos_weight=pos_weight_tensor,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=args.lr_factor,
            patience=args.lr_patience,
        )
    )
    scaler = make_grad_scaler(
        amp_enabled
    )

    args_dict = vars(args).copy()
    args_dict["split_file"] = str(split_file)

    save_json(
        args_dict,
        output_dir / "train_args.json",
    )
    save_json(
        metadata,
        output_dir / "data_metadata.json",
    )

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print("device:", device)
    print("parameters:", f"{parameter_count:,}")
    print("metadata:", metadata)
    print(
        "effective batch size:",
        args.batch_size
        * args.grad_accum_steps,
    )
    print(
        "spectral kernels:",
        args.spectral_time_kernels,
        "x",
        args.spectral_frequency_kernel,
        "| stride=(1,1)",
    )

    history_path = output_dir / "history.csv"
    if history_path.exists():
        history_path.unlink()

    best_score = (
        float("inf")
        if args.monitor == "val_loss"
        else -float("inf")
    )
    best_epoch = 0
    no_improvement = 0

    for epoch in range(
        1,
        args.max_epochs + 1,
    ):
        (
            train_loss,
            train_target,
            train_probability,
        ) = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            device=device,
            optimizer=optimizer,
            scaler=scaler,
            amp_enabled=amp_enabled,
            grad_accum_steps=(
                args.grad_accum_steps
            ),
        )

        (
            val_loss,
            val_target,
            val_probability,
        ) = run_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            optimizer=None,
            scaler=scaler,
            amp_enabled=amp_enabled,
            grad_accum_steps=1,
        )

        thresholds = np.zeros(
            2,
            dtype=np.float64,
        )

        for channel in range(2):
            (
                thresholds[channel],
                _,
            ) = choose_threshold(
                val_target[:, channel],
                val_probability[:, channel],
                metric=args.threshold_metric,
            )

        train_metrics = evaluate_joint(
            train_target[None, ...],
            train_probability[None, ...],
            thresholds,
        )
        val_metrics = evaluate_joint(
            val_target[None, ...],
            val_probability[None, ...],
            thresholds,
        )

        scheduler.step(val_loss)
        learning_rate = (
            optimizer.param_groups[0]["lr"]
        )

        current_score = (
            val_loss
            if args.monitor == "val_loss"
            else float(
                val_metrics[args.monitor]
            )
        )

        improved = (
            current_score < best_score
            if args.monitor == "val_loss"
            else current_score > best_score
        )

        checkpoint = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "thresholds": thresholds,
            "args": args_dict,
            "metadata": metadata,
            "parameter_count": parameter_count,
            "monitor": args.monitor,
            "monitor_score": current_score,
            "val_loss": val_loss,
            "val_metrics": val_metrics,
        }

        torch.save(
            checkpoint,
            output_dir / "last.pt",
        )

        if improved:
            best_score = current_score
            best_epoch = epoch
            no_improvement = 0

            torch.save(
                checkpoint,
                output_dir / "best.pt",
            )
            save_json(
                val_metrics,
                output_dir
                / "best_validation_metrics.json",
            )
        else:
            no_improvement += 1

        append_history(
            history_path,
            {
                "epoch": epoch,
                "lr": learning_rate,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_mean_accuracy": (
                    train_metrics["mean_accuracy"]
                ),
                "train_mean_f1": (
                    train_metrics["mean_f1"]
                ),
                "val_mean_accuracy": (
                    val_metrics["mean_accuracy"]
                ),
                "val_mean_f1": (
                    val_metrics["mean_f1"]
                ),
                "val_exact_match_accuracy": (
                    val_metrics[
                        "exact_match_accuracy"
                    ]
                ),
                "val_i_accuracy": (
                    val_metrics["I"]["accuracy"]
                ),
                "val_i_ppv": (
                    val_metrics["I"]["precision"]
                ),
                "val_i_sensitivity": (
                    val_metrics["I"]["sensitivity"]
                ),
                "val_i_f1": (
                    val_metrics["I"]["f1"]
                ),
                "val_e_accuracy": (
                    val_metrics["E"]["accuracy"]
                ),
                "val_e_ppv": (
                    val_metrics["E"]["precision"]
                ),
                "val_e_sensitivity": (
                    val_metrics["E"]["sensitivity"]
                ),
                "val_e_f1": (
                    val_metrics["E"]["f1"]
                ),
                "threshold_i": thresholds[0],
                "threshold_e": thresholds[1],
            },
        )

        print(
            f"[{epoch:03d}/{args.max_epochs:03d}] "
            f"train_loss={train_loss:.6f} "
            f"val_loss={val_loss:.6f} "
            f"val_mean_acc="
            f"{val_metrics['mean_accuracy']:.4f} "
            f"val_mean_f1="
            f"{val_metrics['mean_f1']:.4f} "
            f"I_f1={val_metrics['I']['f1']:.4f} "
            f"E_f1={val_metrics['E']['f1']:.4f} "
            f"I_thr={thresholds[0]:.2f} "
            f"E_thr={thresholds[1]:.2f} "
            f"lr={learning_rate:.2e}"
        )

        if (
            no_improvement
            >= args.early_stopping
        ):
            print(
                "Early stopping after "
                f"{args.early_stopping} epochs "
                "without improvement."
            )
            break

    print("best epoch:", best_epoch)
    print("best score:", best_score)
    print(
        "checkpoint:",
        output_dir / "best.pt",
    )


if __name__ == "__main__":
    main()

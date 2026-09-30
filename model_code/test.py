from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import datasets_from_split
from metrics import CATEGORY_NAMES, evaluate_joint
from model import build_model_from_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Test the full-resolution spectral Transformer joint I/E model."
        )
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--split-file", default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
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


def safe_torch_load(
    path: Path,
):
    try:
        return torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        return torch.load(
            path,
            map_location="cpu",
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


def read_segment_seconds(
    h5_path: str,
    label_key: str,
) -> float:
    with h5py.File(h5_path, "r") as h5:
        if label_key == "frame_target":
            return float(
                h5.attrs.get(
                    "feature_frame_seconds",
                    0.016,
                )
            )
        return float(
            h5.attrs.get(
                "output_segment_seconds",
                0.032,
            )
        )


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    amp_enabled = bool(
        args.amp and device.type == "cuda"
    )

    checkpoint_path = Path(
        args.checkpoint
    ).resolve()
    checkpoint = safe_torch_load(
        checkpoint_path
    )

    train_args = checkpoint["args"]
    metadata = checkpoint["metadata"]

    split_file = (
        Path(args.split_file).resolve()
        if args.split_file
        else Path(
            train_args["split_file"]
        ).resolve()
    )

    if not split_file.exists():
        fallback = (
            checkpoint_path.parent
            / "split_indices.npz"
        )
        if fallback.exists():
            split_file = fallback
        else:
            raise FileNotFoundError(
                f"Split file not found: {split_file}"
            )

    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else checkpoint_path.parent / "test"
    )
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    _, _, test_set, _ = datasets_from_split(
        split_file,
        input_key=train_args.get(
            "input_key",
            "features",
        ),
        label_key=train_args.get(
            "label_key",
            "frame_target",
        ),
    )

    loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(
            args.num_workers > 0
        ),
    )

    model = build_model_from_config(
        config=train_args,
        feature_dim=int(
            metadata["feature_dim"]
        ),
    )
    model.load_state_dict(
        checkpoint["model_state"],
        strict=True,
    )
    model.to(device).eval()

    thresholds = np.asarray(
        checkpoint["thresholds"],
        dtype=np.float64,
    )

    all_probability: list[np.ndarray] = []
    all_target: list[np.ndarray] = []
    all_category: list[int] = []
    all_source: list[int] = []
    all_index: list[int] = []

    with torch.no_grad():
        for batch in loader:
            (
                features,
                target,
                input_lengths,
                category,
                source,
                source_index,
            ) = batch

            features = features.to(
                device,
                non_blocking=True,
            )
            input_lengths = input_lengths.to(
                device,
                non_blocking=True,
            )

            with torch.autocast(
                device_type=device.type,
                enabled=amp_enabled,
            ):
                logits, output_lengths = model(
                    features,
                    input_lengths,
                )

            probability = torch.sigmoid(
                logits
            ).cpu().numpy()
            target_np = target.numpy()
            lengths_np = (
                output_lengths.cpu().numpy()
            )

            for row, length in enumerate(
                lengths_np
            ):
                length = int(length)
                all_probability.append(
                    probability[row, :length]
                )
                all_target.append(
                    target_np[row, :length]
                )
                all_category.append(
                    int(category[row])
                )
                all_source.append(
                    int(source[row])
                )
                all_index.append(
                    int(source_index[row])
                )

    probabilities = np.stack(
        all_probability,
        axis=0,
    )
    targets = np.stack(
        all_target,
        axis=0,
    )
    categories = np.asarray(
        all_category,
        dtype=np.int8,
    )
    sources = np.asarray(
        all_source,
        dtype=np.int8,
    )
    source_indices = np.asarray(
        all_index,
        dtype=np.int64,
    )

    segment_seconds = read_segment_seconds(
        test_set.paths[0],
        train_args.get(
            "label_key",
            "frame_target",
        ),
    )

    metrics = evaluate_joint(
        targets=targets,
        probabilities=probabilities,
        thresholds=thresholds,
        categories=categories,
    )
    metrics["segment_seconds"] = segment_seconds
    metrics["label_key"] = train_args.get(
        "label_key",
        "frame_target",
    )

    save_json(
        metrics,
        output_dir / "metrics.json",
    )

    np.savez_compressed(
        output_dir / "predictions.npz",
        probabilities=probabilities.astype(
            np.float32
        ),
        targets=targets.astype(np.float32),
        categories=categories.astype(
            np.int8
        ),
        sources=sources.astype(np.int8),
        source_indices=source_indices.astype(
            np.int64
        ),
        thresholds=thresholds.astype(
            np.float32
        ),
        segment_seconds=np.asarray(
            segment_seconds,
            dtype=np.float32,
        ),
        label_key=np.asarray(
            train_args.get(
                "label_key",
                "frame_target",
            )
        ),
    )

    prediction_binary = (
        probabilities
        >= thresholds.reshape(1, 1, 2)
    )
    target_binary = targets > 0.5

    with (
        output_dir / "record_summary.csv"
    ).open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        fieldnames = [
            "row",
            "source_code",
            "source_index",
            "category",
            "source_file",
            "subject_id",
            "i_segment_accuracy",
            "e_segment_accuracy",
            "exact_match_accuracy",
            "predicted_overlap_rate",
            "true_overlap_rate",
        ]
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )
        writer.writeheader()

        for row in range(len(test_set)):
            info = test_set.metadata(row)

            writer.writerow(
                {
                    "row": row,
                    "source_code": int(
                        sources[row]
                    ),
                    "source_index": int(
                        source_indices[row]
                    ),
                    "category": CATEGORY_NAMES[
                        int(categories[row])
                    ],
                    "source_file": info[
                        "source_file"
                    ],
                    "subject_id": info[
                        "subject_id"
                    ],
                    "i_segment_accuracy": float(
                        np.mean(
                            prediction_binary[
                                row,
                                :,
                                0,
                            ]
                            == target_binary[
                                row,
                                :,
                                0,
                            ]
                        )
                    ),
                    "e_segment_accuracy": float(
                        np.mean(
                            prediction_binary[
                                row,
                                :,
                                1,
                            ]
                            == target_binary[
                                row,
                                :,
                                1,
                            ]
                        )
                    ),
                    "exact_match_accuracy": float(
                        np.mean(
                            np.all(
                                prediction_binary[row]
                                == target_binary[row],
                                axis=-1,
                            )
                        )
                    ),
                    "predicted_overlap_rate": float(
                        np.mean(
                            np.logical_and(
                                prediction_binary[
                                    row,
                                    :,
                                    0,
                                ],
                                prediction_binary[
                                    row,
                                    :,
                                    1,
                                ],
                            )
                        )
                    ),
                    "true_overlap_rate": float(
                        np.mean(
                            np.logical_and(
                                target_binary[
                                    row,
                                    :,
                                    0,
                                ],
                                target_binary[
                                    row,
                                    :,
                                    1,
                                ],
                            )
                        )
                    ),
                }
            )

    print(
        json.dumps(
            metrics,
            indent=2,
            ensure_ascii=False,
        )
    )
    print(
        "predictions:",
        output_dir / "predictions.npz",
    )
    print(
        "record summary:",
        output_dir / "record_summary.csv",
    )


if __name__ == "__main__":
    main()

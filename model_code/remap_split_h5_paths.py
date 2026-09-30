#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Copy an existing split_indices.npz while replacing only "
            "the train/test HDF5 paths. All train/val/test indices, "
            "groups, categories, and source-file arrays are preserved."
        )
    )
    parser.add_argument("--source-split", required=True, type=Path)
    parser.add_argument("--train-h5", required=True, type=Path)
    parser.add_argument("--test-h5", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--label-key",
        default="frame_target",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    source_split = args.source_split.resolve()
    train_h5 = args.train_h5.resolve()
    test_h5 = args.test_h5.resolve()
    output = args.output.resolve()

    if not source_split.exists():
        raise FileNotFoundError(source_split)
    if not train_h5.exists():
        raise FileNotFoundError(train_h5)
    if not test_h5.exists():
        raise FileNotFoundError(test_h5)
    if output.exists() and not args.overwrite:
        raise FileExistsError(
            f"{output} exists. Pass --overwrite to replace it."
        )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with np.load(
        source_split,
        allow_pickle=False,
    ) as source:
        payload = {
            key: source[key]
            for key in source.files
        }

    original_train_h5 = str(
        payload["train_h5"].item()
    )
    original_test_h5 = str(
        payload["test_h5"].item()
    )

    payload["train_h5"] = np.asarray(
        str(train_h5)
    )
    payload["test_h5"] = np.asarray(
        str(test_h5)
    )
    payload["label_key"] = np.asarray(
        args.label_key
    )

    np.savez_compressed(
        output,
        **payload,
    )

    summary = {
        "source_split": str(source_split),
        "output_split": str(output),
        "original_train_h5": original_train_h5,
        "original_test_h5": original_test_h5,
        "new_train_h5": str(train_h5),
        "new_test_h5": str(test_h5),
        "label_key": args.label_key,
        "preserved_keys": sorted(payload.keys()),
        "counts": {
            "train": int(
                len(payload["train_index"])
            ),
            "validation": int(
                len(payload["val_index"])
            ),
            "test": int(
                len(payload["test_index"])
            ),
        },
    }

    output.with_suffix(".json").write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

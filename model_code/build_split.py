from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import h5py
import numpy as np


CATEGORY_NAMES = np.asarray(
    ["background_only", "i_only", "e_only", "both_ie"],
    dtype=object,
)


def decode_string(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def classify_presence(has_i: np.ndarray, has_e: np.ndarray) -> np.ndarray:
    category = np.zeros(len(has_i), dtype=np.int8)
    category[has_i & ~has_e] = 1
    category[~has_i & has_e] = 2
    category[has_i & has_e] = 3
    return category


def scan_h5(
    path: str | Path,
    source_code: int,
    label_key: str = "frame_target",
    group_key: str | None = "subject_id",
    chunk_size: int = 512,
) -> dict[str, np.ndarray]:
    path = Path(path).resolve()
    rows: dict[str, list[Any]] = {
        "source": [],
        "index": [],
        "category": [],
        "group": [],
        "source_file": [],
        "overlap_frames": [],
    }

    with h5py.File(path, "r") as h5:
        if label_key not in h5:
            raise KeyError(f"{path}: missing label dataset {label_key!r}")

        labels = h5[label_key]
        if labels.ndim != 3 or labels.shape[-1] < 2:
            raise ValueError(
                f"{path}: expected {label_key} shape [N,T,C>=2], got {labels.shape}"
            )

        n_samples = len(labels)
        source_file_ds = h5.get("source_file")
        group_ds = h5.get(group_key) if group_key else None

        for start in range(0, n_samples, chunk_size):
            end = min(start + chunk_size, n_samples)
            batch = np.asarray(labels[start:end, :, :2])

            i_mask = batch[..., 0] > 0
            e_mask = batch[..., 1] > 0
            has_i = i_mask.any(axis=1)
            has_e = e_mask.any(axis=1)
            category = classify_presence(has_i, has_e)
            overlap_frames = np.logical_and(i_mask, e_mask).sum(axis=1)

            for local, sample_index in enumerate(range(start, end)):
                source_file = (
                    decode_string(source_file_ds[sample_index])
                    if source_file_ds is not None
                    else f"source{source_code}:{sample_index}"
                )
                group = (
                    decode_string(group_ds[sample_index])
                    if group_ds is not None
                    else source_file
                )
                if not group:
                    group = source_file

                rows["source"].append(source_code)
                rows["index"].append(sample_index)
                rows["category"].append(int(category[local]))
                rows["group"].append(group)
                rows["source_file"].append(source_file)
                rows["overlap_frames"].append(int(overlap_frames[local]))

    return {
        "source": np.asarray(rows["source"], dtype=np.int8),
        "index": np.asarray(rows["index"], dtype=np.int64),
        "category": np.asarray(rows["category"], dtype=np.int8),
        "group": np.asarray(rows["group"], dtype=object),
        "source_file": np.asarray(rows["source_file"], dtype=object),
        "overlap_frames": np.asarray(rows["overlap_frames"], dtype=np.int32),
    }


def grouped_train_val_split(
    groups: np.ndarray,
    val_ratio: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio must be between 0 and 1.")

    unique_groups, inverse, counts = np.unique(
        groups.astype(str),
        return_inverse=True,
        return_counts=True,
    )
    if len(unique_groups) < 2:
        raise ValueError(
            "At least two distinct groups are required for train/validation split."
        )

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(unique_groups))
    target_val = max(1, int(round(len(groups) * val_ratio)))

    selected_val_groups: list[int] = []
    selected_count = 0

    for group_id in order:
        remaining_groups = len(unique_groups) - len(selected_val_groups)
        if remaining_groups <= 1:
            break
        selected_val_groups.append(int(group_id))
        selected_count += int(counts[group_id])
        if selected_count >= target_val:
            break

    val_group_mask = np.isin(inverse, selected_val_groups)
    val_local = np.flatnonzero(val_group_mask)
    train_local = np.flatnonzero(~val_group_mask)

    if len(train_local) == 0 or len(val_local) == 0:
        raise RuntimeError("Failed to create non-empty train and validation splits.")

    return train_local, val_local


def build_split(
    train_h5: str | Path,
    test_h5: str | Path,
    output_path: str | Path,
    val_ratio: float = 0.15,
    seed: int = 42,
    label_key: str = "frame_target",
    group_key: str | None = "subject_id",
    chunk_size: int = 512,
) -> dict[str, Any]:
    train_h5 = Path(train_h5).resolve()
    test_h5 = Path(test_h5).resolve()
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    first = scan_h5(
        train_h5,
        source_code=0,
        label_key=label_key,
        group_key=group_key,
        chunk_size=chunk_size,
    )
    second = scan_h5(
        test_h5,
        source_code=1,
        label_key=label_key,
        group_key=group_key,
        chunk_size=chunk_size,
    )

    pool = {
        key: np.concatenate([first[key], second[key]])
        for key in first
    }

    both_pool = np.flatnonzero(pool["category"] == 3)
    non_both_pool = np.flatnonzero(pool["category"] != 3)

    if len(both_pool) < 2:
        raise RuntimeError(
            f"Only {len(both_pool)} both-I/E samples were found; cannot train."
        )
    if len(non_both_pool) == 0:
        raise RuntimeError("No non-both-I/E samples were found for test.")

    train_local, val_local = grouped_train_val_split(
        pool["group"][both_pool],
        val_ratio=val_ratio,
        seed=seed,
    )
    train_pool = both_pool[train_local]
    val_pool = both_pool[val_local]
    test_pool = non_both_pool

    np.savez_compressed(
        output_path,
        train_source=pool["source"][train_pool],
        train_index=pool["index"][train_pool],
        val_source=pool["source"][val_pool],
        val_index=pool["index"][val_pool],
        test_source=pool["source"][test_pool],
        test_index=pool["index"][test_pool],
        test_category=pool["category"][test_pool],
        train_group=pool["group"][train_pool].astype(str),
        val_group=pool["group"][val_pool].astype(str),
        test_group=pool["group"][test_pool].astype(str),
        train_source_file=pool["source_file"][train_pool].astype(str),
        val_source_file=pool["source_file"][val_pool].astype(str),
        test_source_file=pool["source_file"][test_pool].astype(str),
        train_h5=np.asarray(str(train_h5)),
        test_h5=np.asarray(str(test_h5)),
        label_key=np.asarray(label_key),
        group_key=np.asarray("" if group_key is None else group_key),
        category_names=CATEGORY_NAMES.astype(str),
        seed=np.asarray(seed, dtype=np.int64),
        val_ratio=np.asarray(val_ratio, dtype=np.float64),
    )

    train_groups = set(pool["group"][train_pool].astype(str))
    val_groups = set(pool["group"][val_pool].astype(str))
    test_groups = set(pool["group"][test_pool].astype(str))

    category_counter = Counter(
        CATEGORY_NAMES[pool["category"][test_pool]].tolist()
    )
    source_counter = Counter(pool["source"][test_pool].tolist())

    summary = {
        "train_h5": str(train_h5),
        "test_h5": str(test_h5),
        "split_file": str(output_path),
        "rule": {
            "train_val_candidates": "recordings containing at least one I frame and at least one E frame",
            "test": "recordings not containing both I and E",
            "validation": "group-disjoint subset of both-I/E candidates",
        },
        "counts": {
            "all": int(len(pool["index"])),
            "both_ie_all": int(len(both_pool)),
            "non_both_test": int(len(test_pool)),
            "train": int(len(train_pool)),
            "validation": int(len(val_pool)),
            "test": int(len(test_pool)),
        },
        "test_category_counts": {
            name: int(category_counter.get(name, 0))
            for name in CATEGORY_NAMES.tolist()
        },
        "test_original_h5_counts": {
            "original_train_h5": int(source_counter.get(0, 0)),
            "original_test_h5": int(source_counter.get(1, 0)),
        },
        "group_overlap": {
            "train_val": int(len(train_groups & val_groups)),
            "train_test": int(len(train_groups & test_groups)),
            "val_test": int(len(val_groups & test_groups)),
        },
        "warning": (
            "The requested composition split can place the same subject in "
            "train and test when that subject has recordings in different "
            "composition categories. Check group_overlap before interpreting results."
        ),
    }

    summary_path = output_path.with_suffix(".json")
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build composition split for joint I/E training."
    )
    parser.add_argument("--train-h5", required=True)
    parser.add_argument("--test-h5", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--label-key", default="frame_target")
    parser.add_argument("--group-key", default="subject_id")
    parser.add_argument("--chunk-size", type=int, default=512)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_split(
        train_h5=args.train_h5,
        test_h5=args.test_h5,
        output_path=args.output,
        val_ratio=args.val_ratio,
        seed=args.seed,
        label_key=args.label_key,
        group_key=args.group_key or None,
        chunk_size=args.chunk_size,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

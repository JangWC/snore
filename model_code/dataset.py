from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


SOURCE_NAMES = {0: "original_train_h5", 1: "original_test_h5"}
CATEGORY_NAMES = {
    0: "background_only",
    1: "i_only",
    2: "e_only",
    3: "both_ie",
}


def decode_string(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def load_split_file(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(Path(path).resolve(), allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


class JointIEH5Dataset(Dataset):
    def __init__(
        self,
        train_h5: str | Path,
        test_h5: str | Path,
        sources: np.ndarray,
        indices: np.ndarray,
        categories: np.ndarray | None = None,
        input_key: str = "features",
        label_key: str = "target",
    ) -> None:
        self.paths = {
            0: str(Path(train_h5).resolve()),
            1: str(Path(test_h5).resolve()),
        }
        self.sources = np.asarray(sources, dtype=np.int8)
        self.indices = np.asarray(indices, dtype=np.int64)
        self.categories = (
            np.full(len(self.indices), 3, dtype=np.int8)
            if categories is None
            else np.asarray(categories, dtype=np.int8)
        )
        self.input_key = input_key
        self.label_key = label_key
        self._handles: dict[int, h5py.File] = {}

        if not (
            len(self.sources)
            == len(self.indices)
            == len(self.categories)
        ):
            raise ValueError("sources, indices, and categories must have equal length.")

        with h5py.File(self.paths[0], "r") as h5:
            self.input_shape = tuple(h5[self.input_key].shape[1:])
            self.target_shape = tuple(h5[self.label_key].shape[1:])
        if len(self.input_shape) != 2:
            raise ValueError(
                f"Expected input [T,F], got {self.input_shape}"
            )
        if len(self.target_shape) != 2 or self.target_shape[-1] < 2:
            raise ValueError(
                f"Expected target [T,C>=2], got {self.target_shape}"
            )

    def _handle(self, source: int) -> h5py.File:
        if source not in self._handles:
            self._handles[source] = h5py.File(
                self.paths[source],
                "r",
            )
        return self._handles[source]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int):
        source = int(self.sources[item])
        index = int(self.indices[item])
        h5 = self._handle(source)

        features = np.asarray(
            h5[self.input_key][index],
            dtype=np.float32,
        )
        target = np.asarray(
            h5[self.label_key][index, :, :2],
            dtype=np.float32,
        )

        return (
            torch.from_numpy(features),
            torch.from_numpy(target),
            torch.tensor(features.shape[0], dtype=torch.long),
            torch.tensor(int(self.categories[item]), dtype=torch.long),
            torch.tensor(source, dtype=torch.long),
            torch.tensor(index, dtype=torch.long),
        )

    def metadata(self, item: int) -> dict[str, Any]:
        source = int(self.sources[item])
        index = int(self.indices[item])
        h5 = self._handle(source)

        return {
            "source_code": source,
            "source_name": SOURCE_NAMES[source],
            "source_index": index,
            "category_id": int(self.categories[item]),
            "category": CATEGORY_NAMES[int(self.categories[item])],
            "source_file": (
                decode_string(h5["source_file"][index])
                if "source_file" in h5
                else f"{SOURCE_NAMES[source]}:{index}"
            ),
            "subject_id": (
                decode_string(h5["subject_id"][index])
                if "subject_id" in h5
                else ""
            ),
        }

    def close(self) -> None:
        for handle in self._handles.values():
            try:
                handle.close()
            except Exception:
                pass
        self._handles.clear()

    def __del__(self) -> None:
        self.close()


def datasets_from_split(
    split_file: str | Path,
    input_key: str = "features",
    label_key: str = "target",
) -> tuple[JointIEH5Dataset, JointIEH5Dataset, JointIEH5Dataset, dict[str, Any]]:
    split = load_split_file(split_file)
    train_h5 = str(split["train_h5"].item())
    test_h5 = str(split["test_h5"].item())

    train = JointIEH5Dataset(
        train_h5=train_h5,
        test_h5=test_h5,
        sources=split["train_source"],
        indices=split["train_index"],
        categories=np.full(len(split["train_index"]), 3, dtype=np.int8),
        input_key=input_key,
        label_key=label_key,
    )
    val = JointIEH5Dataset(
        train_h5=train_h5,
        test_h5=test_h5,
        sources=split["val_source"],
        indices=split["val_index"],
        categories=np.full(len(split["val_index"]), 3, dtype=np.int8),
        input_key=input_key,
        label_key=label_key,
    )
    test = JointIEH5Dataset(
        train_h5=train_h5,
        test_h5=test_h5,
        sources=split["test_source"],
        indices=split["test_index"],
        categories=split["test_category"],
        input_key=input_key,
        label_key=label_key,
    )

    metadata = {
        "train_h5": train_h5,
        "test_h5": test_h5,
        "input_shape": train.input_shape,
        "target_shape": train.target_shape,
        "feature_dim": int(train.input_shape[-1]),
        "input_frames": int(train.input_shape[0]),
        "output_frames": int(train.target_shape[0]),
        "train_samples": len(train),
        "val_samples": len(val),
        "test_samples": len(test),
    }
    return train, val, test, metadata


def compute_pos_weight(
    dataset: JointIEH5Dataset,
    chunk_size: int = 256,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    positive = np.zeros(2, dtype=np.float64)
    total = np.zeros(2, dtype=np.float64)

    for start in range(0, len(dataset), chunk_size):
        end = min(start + chunk_size, len(dataset))
        grouped: dict[int, list[int]] = {0: [], 1: []}
        for item in range(start, end):
            grouped[int(dataset.sources[item])].append(item)

        for source, items in grouped.items():
            if not items:
                continue
            h5 = dataset._handle(source)
            h5_indices = dataset.indices[items]
            order = np.argsort(h5_indices)
            sorted_indices = h5_indices[order]
            labels = np.asarray(
                h5[dataset.label_key][sorted_indices, :, :2],
                dtype=np.float32,
            )
            positive += labels.sum(axis=(0, 1))
            total += np.prod(labels.shape[:2])

    negative = total - positive
    pos_weight = negative / np.maximum(positive, 1.0)
    return pos_weight.astype(np.float32), positive, negative

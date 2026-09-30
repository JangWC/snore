#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import torch

from ie_inference import IEInferencePipeline


def decode(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def metrics(a: np.ndarray, b: np.ndarray):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    diff = a - b
    return {
        "shape_a": tuple(a.shape),
        "shape_b": tuple(b.shape),
        "max_abs": float(np.max(np.abs(diff))),
        "mean_abs": float(np.mean(np.abs(diff))),
        "rmse": float(np.sqrt(np.mean(diff * diff))),
        "exact_equal_rate": float(np.mean(a == b)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare raw-WAV preprocessing/model output with one stored HDF5 sample."
        )
    )
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--stats-h5", required=True, type=Path)
    parser.add_argument("--raw-root", required=True, type=Path)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--project-model", type=Path, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--feature-mode",
        choices=("auto", "mel129", "stft129"),
        default="auto",
    )
    args = parser.parse_args()

    pipeline = IEInferencePipeline(
        run_dir=args.run_dir,
        stats_h5=args.stats_h5,
        checkpoint=args.checkpoint,
        project_model=args.project_model,
        device=args.device,
        feature_mode=args.feature_mode,
        crop_mode="start",
    )

    with h5py.File(args.stats_h5, "r") as h5:
        if "source_file" not in h5:
            raise KeyError("HDF5에 source_file dataset이 없습니다.")
        if "features" not in h5:
            raise KeyError("HDF5에 features dataset이 없습니다.")

        source_file = decode(h5["source_file"][args.index])
        stored_feature = np.asarray(
            h5["features"][args.index],
            dtype=np.float16,
        )

    raw_path = args.raw_root / source_file
    if not raw_path.is_file():
        raise FileNotFoundError(
            f"raw WAV를 찾지 못했습니다: {raw_path}"
        )

    prep = pipeline.preprocessor.preprocess_file(
        raw_path,
        crop_mode="start",
    )
    recomputed_feature = (
        prep["standardized_features_f16"]
        .detach()
        .cpu()
        .numpy()
    )

    print("\n[FEATURE COMPARISON]")
    print("source:", source_file)
    for key, value in metrics(
        stored_feature,
        recomputed_feature,
    ).items():
        print(f"{key}: {value}")

    stored_tensor = torch.from_numpy(
        stored_feature.astype(np.float32)
    )
    recomputed_tensor = torch.from_numpy(
        recomputed_feature.astype(np.float32)
    )

    stored_i, stored_e, stored_len = (
        pipeline.infer_feature_tensor(stored_tensor)
    )
    recomputed_i, recomputed_e, recomputed_len = (
        pipeline.infer_feature_tensor(recomputed_tensor)
    )

    print("\n[MODEL OUTPUT COMPARISON]")
    print("stored_output_length:", stored_len)
    print("recomputed_output_length:", recomputed_len)
    print("I:", metrics(stored_i, recomputed_i))
    print("E:", metrics(stored_e, recomputed_e))

    feature_report = metrics(
        stored_feature,
        recomputed_feature,
    )
    if feature_report["max_abs"] == 0.0:
        print("\n[PASS] Recomputed HDF5 feature is bit-identical.")
    else:
        print(
            "\n[FAIL] Preprocessing differs from stored HDF5. "
            "Check stats H5, feature mode, clipping threshold, and raw WAV."
        )


if __name__ == "__main__":
    main()

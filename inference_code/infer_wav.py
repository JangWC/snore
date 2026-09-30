#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from ie_inference import IEInferencePipeline
from ie_inference.preprocessing import DEFAULT_STATS_PATH


BUNDLE_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RUN_DIR = BUNDLE_ROOT / "checkpoint"
DEFAULT_PROJECT_MODEL = BUNDLE_ROOT / "model_code" / "model.py"


def load_optional_labels(
    path: Optional[Path],
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    if path is None:
        return None, None

    path = Path(path)
    if path.suffix.lower() == ".npz":
        data = np.load(path, allow_pickle=False)
        i = next(
            (
                np.asarray(data[key])
                for key in (
                    "i",
                    "I",
                    "true_i",
                    "i_label",
                    "i_labels",
                )
                if key in data
            ),
            None,
        )
        e = next(
            (
                np.asarray(data[key])
                for key in (
                    "e",
                    "E",
                    "true_e",
                    "e_label",
                    "e_labels",
                )
                if key in data
            ),
            None,
        )
        if i is None or e is None:
            raise KeyError(
                "label npz에는 I/E 배열이 필요합니다."
            )
        return i, e

    arr = np.asarray(
        np.load(path, allow_pickle=False)
    )
    if arr.ndim != 2:
        raise ValueError(
            "label npy shape는 [2,T] 또는 [T,2]이어야 합니다."
        )
    if arr.shape[0] == 2:
        return arr[0], arr[1]
    if arr.shape[1] == 2:
        return arr[:, 0], arr[:, 1]
    raise ValueError(
        "label npy에서 I/E 축(size=2)을 찾지 못했습니다."
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "External WAV -> exact TRAIN preprocessing -> "
            "notebook-identical joint I/E inference"
        )
    )
    parser.add_argument(
        "--wav",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=DEFAULT_RUN_DIR,
    )
    parser.add_argument(
        "--stats-h5",
        type=Path,
        default=DEFAULT_STATS_PATH,
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--project-model",
        type=Path,
        default=DEFAULT_PROJECT_MODEL,
        help="번들에 포함된 model_code/model.py를 기본으로 사용",
    )
    # Kept only so old commands do not fail.
    parser.add_argument(
        "--model-factory",
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--input-layout",
        choices=("btf",),
        default="btf",
        help="notebook과 동일하게 [B,T,F]만 지원",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./inference_outputs"),
    )
    parser.add_argument(
        "--device",
        default=None,
        help="cuda:0 또는 cpu",
    )
    parser.add_argument(
        "--feature-mode",
        choices=("auto", "mel129", "stft129"),
        default="auto",
    )
    parser.add_argument(
        "--crop-mode",
        choices=("start", "center"),
        default="start",
    )
    parser.add_argument(
        "--i-threshold",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--e-threshold",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--label",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--min-segment-sec",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--no-figure",
        action="store_true",
    )
    parser.add_argument(
        "--no-npz",
        action="store_true",
    )
    parser.add_argument(
        "--no-csv",
        action="store_true",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    true_i, true_e = load_optional_labels(
        args.label
    )

    pipeline = IEInferencePipeline(
        run_dir=args.run_dir,
        stats_h5=args.stats_h5,
        checkpoint=args.checkpoint,
        project_model=args.project_model,
        device=args.device,
        feature_mode=args.feature_mode,
        crop_mode=args.crop_mode,
        i_threshold=args.i_threshold,
        e_threshold=args.e_threshold,
        model_factory=args.model_factory,
        input_layout=args.input_layout,
    )

    outputs = pipeline.run(
        wav_path=args.wav,
        output_dir=args.output_dir,
        save_figure=not args.no_figure,
        save_npz=not args.no_npz,
        save_csv=not args.no_csv,
        min_segment_sec=args.min_segment_sec,
        true_i=true_i,
        true_e=true_e,
    )

    print("[DONE]")
    for name, path in outputs.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()

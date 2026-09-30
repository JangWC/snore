from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from .model_loader import load_training_model
from .preprocessing import (
    DEFAULT_STATS_H5,
    ExactFeatureExtractor,
    load_training_statistics,
)
from .visualization import compute_linear_spectrogram, save_inference_figure


def _extract_segments(
    times: np.ndarray,
    mask: np.ndarray,
    label: str,
    min_duration: float = 0.0,
):
    times = np.asarray(times, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    if len(times) == 0:
        return []

    dt = float(np.median(np.diff(times))) if len(times) > 1 else 0.0
    padded = np.pad(mask.astype(np.int8), (1, 1))
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)

    rows = []
    for start_idx, end_idx in zip(starts, ends):
        start_time = float(times[start_idx])
        end_time = float(times[end_idx - 1] + dt)
        if end_time - start_time >= min_duration:
            rows.append(
                {
                    "label": label,
                    "start_sec": start_time,
                    "end_sec": end_time,
                    "duration_sec": end_time - start_time,
                }
            )
    return rows


class IEInferencePipeline:
    def __init__(
        self,
        run_dir: Path,
        stats_h5: Path = DEFAULT_STATS_H5,
        checkpoint: Optional[Path] = None,
        project_model: Optional[Path] = None,
        device: Optional[str] = None,
        feature_mode: str = "auto",
        crop_mode: str = "start",
        i_threshold: Optional[float] = None,
        e_threshold: Optional[float] = None,
        model_factory: Optional[str] = None,
        input_layout: Optional[str] = None,
    ) -> None:
        self.run_dir = Path(run_dir).expanduser().resolve()
        if not self.run_dir.is_dir():
            raise NotADirectoryError(f"run_dir가 없습니다: {self.run_dir}")

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device_obj = torch.device(self.device)
        self.crop_mode = crop_mode

        if model_factory is not None:
            print(
                "[WARN] --model-factory는 v4에서 사용하지 않습니다. "
                "notebook과 동일하게 학습 프로젝트의 build_model_from_config를 사용합니다."
            )
        if input_layout not in (None, "btf"):
            raise ValueError(
                "notebook 모델 입력은 [B,T,F]로 고정입니다. --input-layout btf를 사용하십시오."
            )

        self.statistics = load_training_statistics(stats_h5, feature_mode)
        self.preprocessor = ExactFeatureExtractor(
            self.statistics,
            device=self.device,
        )
        self.feature_config = self.statistics.config

        loaded = load_training_model(
            run_dir=self.run_dir,
            checkpoint_path=checkpoint,
            project_model_path=project_model,
            device=self.device,
        )
        self.model = loaded.model
        self.checkpoint = loaded.checkpoint
        self.checkpoint_path = loaded.checkpoint_path
        self.train_args = loaded.train_args
        self.metadata = loaded.metadata
        self.project_model_path = loaded.project_model_path

        checkpoint_thresholds = loaded.thresholds
        self.i_threshold = float(
            checkpoint_thresholds[0]
            if i_threshold is None
            else i_threshold
        )
        self.e_threshold = float(
            checkpoint_thresholds[1]
            if e_threshold is None
            else e_threshold
        )

        checkpoint_feature_dim = int(self.metadata["feature_dim"])
        if checkpoint_feature_dim != 193:
            raise ValueError(
                f"checkpoint feature_dim={checkpoint_feature_dim}, expected=193"
            )

        clip_text = (
            "disabled"
            if self.statistics.clip_threshold is None
            else f"{self.statistics.clip_threshold:.8g}"
        )
        print(f"[INFO] stats H5: {self.statistics.path}")
        print(
            "[INFO] preprocessing: "
            f"mode={self.statistics.feature_mode}, "
            f"sr={self.feature_config.target_sr}, "
            f"duration={self.feature_config.record_sec:.1f}s, "
            f"n_fft={self.feature_config.n_fft}, "
            f"win={self.feature_config.win_length}, "
            f"hop={self.feature_config.hop_length}, "
            f"frames={self.feature_config.num_frames}, "
            f"feature_dim={checkpoint_feature_dim}, "
            f"clip={clip_text}"
        )
        print("[INFO] feature storage parity: clip [-12,12] -> float16 quantization -> float32 model input")
        print("[INFO] model input: features [B,T,F] + input_lengths [B]")
        print("[INFO] model output: (logits [B,T,2], output_lengths [B])")
        print(
            f"[INFO] checkpoint thresholds: "
            f"I={checkpoint_thresholds[0]:.6f}, "
            f"E={checkpoint_thresholds[1]:.6f}"
        )
        if (
            self.i_threshold != checkpoint_thresholds[0]
            or self.e_threshold != checkpoint_thresholds[1]
        ):
            print(
                f"[WARN] threshold override: "
                f"I={self.i_threshold:.6f}, "
                f"E={self.e_threshold:.6f}"
            )

    @torch.inference_mode()
    def infer_feature_tensor(
        self,
        feature: torch.Tensor,
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Exact notebook behavior:

            features = features.to(DEVICE)
            input_lengths = input_lengths.to(DEVICE)
            with torch.autocast(...):
                logits, output_lengths = model(features, input_lengths)
            probability = sigmoid(logits)
            probability[row, :output_lengths[row]]
        """
        if tuple(feature.shape) != (938, 193):
            raise ValueError(
                f"Expected feature [938,193], got {tuple(feature.shape)}"
            )

        features = feature.unsqueeze(0).to(
            self.device_obj,
            dtype=torch.float32,
        )
        input_lengths = torch.tensor(
            [feature.shape[0]],
            dtype=torch.long,
            device=self.device_obj,
        )

        with torch.autocast(
            device_type=self.device_obj.type,
            enabled=(self.device_obj.type == "cuda"),
        ):
            output = self.model(
                features,
                input_lengths,
            )

        if not isinstance(output, (tuple, list)) or len(output) != 2:
            raise TypeError(
                "notebook의 model output은 (logits, output_lengths)여야 합니다. "
                f"실제 type={type(output)}"
            )

        logits, output_lengths = output
        if not torch.is_tensor(logits):
            raise TypeError(f"logits가 Tensor가 아닙니다: {type(logits)}")
        if not torch.is_tensor(output_lengths):
            output_lengths = torch.as_tensor(output_lengths)

        if logits.ndim != 3 or logits.shape[0] != 1 or logits.shape[-1] != 2:
            raise ValueError(
                f"Expected logits [1,T,2], got {tuple(logits.shape)}"
            )

        output_length = int(output_lengths.reshape(-1)[0].item())
        if output_length <= 0 or output_length > logits.shape[1]:
            raise ValueError(
                f"Invalid output_length={output_length}, logits T={logits.shape[1]}"
            )

        probabilities = torch.sigmoid(
            logits[0, :output_length]
        ).float().cpu().numpy()

        if probabilities.shape != (output_length, 2):
            raise RuntimeError(
                f"Probability shape error: {probabilities.shape}"
            )

        return (
            probabilities[:, 0].astype(np.float32),
            probabilities[:, 1].astype(np.float32),
            output_length,
        )

    def _probability_times(self, n_outputs: int) -> np.ndarray:
        # Notebook plot uses:
        # time = (arange(len(truth)) + 0.5) * feature_frame_seconds
        frame_seconds = (
            self.feature_config.hop_length
            / self.feature_config.target_sr
        )
        return (
            np.arange(n_outputs, dtype=np.float64) + 0.5
        ) * frame_seconds

    def run(
        self,
        wav_path: Path,
        output_dir: Path,
        save_figure: bool = True,
        save_npz: bool = True,
        save_csv: bool = True,
        min_segment_sec: float = 0.0,
        true_i: Optional[np.ndarray] = None,
        true_e: Optional[np.ndarray] = None,
    ) -> Dict[str, Path]:
        wav_path = Path(wav_path).expanduser().resolve()
        if not wav_path.is_file():
            raise FileNotFoundError(f"WAV 파일이 없습니다: {wav_path}")

        output_dir = Path(output_dir).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        prep = self.preprocessor.preprocess_file(
            wav_path,
            crop_mode=self.crop_mode,
        )
        feature = prep["standardized_features"]

        print(
            "[INFO] audio length: "
            f"{prep['original_duration_sec']:.3f}s -> "
            f"{prep['processed_duration_sec']:.3f}s "
            f"({prep['length_action']}, "
            f"crop_start={prep['crop_start_sec']:.3f}s, "
            f"pad={prep['pad_duration_sec']:.3f}s)"
        )
        print(
            f"[INFO] model input: features=(1,{feature.shape[0]},{feature.shape[1]}), "
            f"input_lengths=[{feature.shape[0]}]"
        )

        i_prob, e_prob, output_length = self.infer_feature_tensor(feature)
        prob_times = self._probability_times(output_length)

        print(
            f"[INFO] logits/probability: [1,{output_length},2]"
        )
        print(
            f"[INFO] probability range: "
            f"I=[{i_prob.min():.6f},{i_prob.max():.6f}], "
            f"E=[{e_prob.min():.6f},{e_prob.max():.6f}]"
        )

        result: Dict[str, Any] = {
            "time_sec": prob_times,
            "i_probability": i_prob,
            "e_probability": e_prob,
            "i_prediction": (
                i_prob >= self.i_threshold
            ).astype(np.uint8),
            "e_prediction": (
                e_prob >= self.e_threshold
            ).astype(np.uint8),
        }

        stem = wav_path.stem
        outputs: Dict[str, Path] = {}

        if save_npz:
            npz_path = output_dir / f"{stem}_ie_inference.npz"
            np.savez_compressed(
                npz_path,
                **result,
                feature_shape=np.asarray(
                    feature.shape,
                    dtype=np.int32,
                ),
                standardized_features=prep[
                    "standardized_features_f16"
                ].detach().cpu().numpy(),
                input_length=np.int32(feature.shape[0]),
                output_length=np.int32(output_length),
                sample_rate=np.int32(
                    self.feature_config.target_sr
                ),
                hop_length=np.int32(
                    self.feature_config.hop_length
                ),
                feature_mode=self.statistics.feature_mode,
                stats_h5=str(self.statistics.path),
                checkpoint=str(self.checkpoint_path),
                project_model=str(self.project_model_path),
                i_threshold=np.float32(self.i_threshold),
                e_threshold=np.float32(self.e_threshold),
                original_duration_sec=np.float32(
                    prep["original_duration_sec"]
                ),
                processed_duration_sec=np.float32(
                    prep["processed_duration_sec"]
                ),
                length_action=str(prep["length_action"]),
                crop_start_sec=np.float32(
                    prep["crop_start_sec"]
                ),
                pad_duration_sec=np.float32(
                    prep["pad_duration_sec"]
                ),
                peak_before_clip=np.float32(
                    prep["peak_before_clip"]
                ),
                clipping_ratio=np.float32(
                    prep["clipping_ratio"]
                ),
            )
            outputs["npz"] = npz_path

        if save_csv:
            rows = []
            rows.extend(
                _extract_segments(
                    prob_times,
                    result["i_prediction"],
                    "I",
                    min_duration=min_segment_sec,
                )
            )
            rows.extend(
                _extract_segments(
                    prob_times,
                    result["e_prediction"],
                    "E",
                    min_duration=min_segment_sec,
                )
            )
            rows.sort(
                key=lambda row: (
                    row["start_sec"],
                    row["label"],
                )
            )

            csv_path = output_dir / f"{stem}_segments.csv"
            with csv_path.open(
                "w",
                encoding="utf-8",
                newline="",
            ) as fp:
                writer = csv.DictWriter(
                    fp,
                    fieldnames=[
                        "label",
                        "start_sec",
                        "end_sec",
                        "duration_sec",
                    ],
                )
                writer.writeheader()
                for row in rows:
                    writer.writerow(
                        {
                            "label": row["label"],
                            "start_sec": (
                                f"{row['start_sec']:.6f}"
                            ),
                            "end_sec": (
                                f"{row['end_sec']:.6f}"
                            ),
                            "duration_sec": (
                                f"{row['duration_sec']:.6f}"
                            ),
                        }
                    )
            outputs["csv"] = csv_path

        if save_figure:
            waveform_t = torch.as_tensor(
                prep["processed_waveform"],
                dtype=torch.float32,
            )
            spec_db, spec_freqs, spec_times = (
                compute_linear_spectrogram(
                    waveform=waveform_t,
                    sample_rate=(
                        self.feature_config.target_sr
                    ),
                    n_fft=self.feature_config.n_fft,
                    hop_length=(
                        self.feature_config.hop_length
                    ),
                    win_length=(
                        self.feature_config.win_length
                    ),
                )
            )

            png_path = output_dir / f"{stem}_ie_inference.png"
            save_inference_figure(
                output_path=png_path,
                wav_name=wav_path.name,
                waveform=waveform_t,
                sample_rate=(
                    self.feature_config.target_sr
                ),
                linear_spec_db=spec_db,
                spec_freqs=spec_freqs,
                spec_times=spec_times,
                feature=prep[
                    "spectral_feature"
                ].detach().cpu().numpy(),
                feature_times=prep["feature_times"],
                feature_title=(
                    "Standardized log-Mel feature (129 bins)"
                    if self.statistics.feature_mode == "mel129"
                    else "Standardized log-power STFT feature (129 bins)"
                ),
                prob_times=prob_times,
                i_prob=i_prob,
                e_prob=e_prob,
                i_threshold=self.i_threshold,
                e_threshold=self.e_threshold,
                true_i=true_i,
                true_e=true_e,
                max_display_hz=2000.0,
            )
            outputs["png"] = png_path

        return outputs

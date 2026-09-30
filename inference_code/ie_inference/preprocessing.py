from __future__ import annotations

import json
import math
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import h5py
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from scipy.signal import butter, resample_poly, sosfiltfilt


DEFAULT_STATS_H5 = Path(
    "/home/tta/Woo_code/data/HF_Lung_V1_pre_joint15_mel129/"
    "HF_Lung_V1_train_15s_mel129_logstd.h5"
)


@dataclass(frozen=True)
class FeatureConfig:
    target_sr: int = 4000
    record_sec: float = 15.0
    highpass_hz: float = 80.0
    highpass_order: int = 10
    n_fft: int = 256
    win_length: int = 256
    hop_length: int = 64
    n_mfcc: int = 20
    spectrogram_n_mels: int = 129
    mfcc_n_mels: int = 40
    delta_width: int = 9

    @property
    def required_samples(self) -> int:
        return int(round(self.record_sec * self.target_sr))

    @property
    def num_frames(self) -> int:
        # torch.stft(center=True): floor(N / hop) + 1
        return self.required_samples // self.hop_length + 1

    @property
    def feature_dim(self) -> int:
        return self.spectrogram_n_mels + 3 * self.n_mfcc + 4


@dataclass(frozen=True)
class TrainingStatistics:
    path: Path
    config: FeatureConfig
    feature_mean: np.ndarray
    feature_std: np.ndarray
    clip_threshold: Optional[float]
    feature_mode: str


def _read_config_json(h5: h5py.File) -> Dict[str, Any]:
    raw = h5.attrs.get("config_json", "{}")
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    try:
        value = json.loads(str(raw))
    except json.JSONDecodeError as exc:
        raise ValueError("stats H5의 config_json을 읽지 못했습니다.") from exc
    return value if isinstance(value, dict) else {}


def _detect_feature_mode(h5: h5py.File, cfg: Dict[str, Any]) -> str:
    if "spectrogram_n_mels" in cfg or "mel_bin_index" in h5:
        return "mel129"
    if "frequencies_hz" in h5:
        return "stft129"

    representation = str(h5.attrs.get("spectrogram_representation", "")).lower()
    if "mel" in representation:
        return "mel129"
    if "log10(power)" in representation or "log-power" in representation:
        return "stft129"
    raise ValueError("stats H5에서 feature_mode를 판별하지 못했습니다.")


def load_training_statistics(
    stats_h5: Path,
    requested_mode: str = "auto",
) -> TrainingStatistics:
    path = Path(stats_h5).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"TRAIN 통계 HDF5를 찾지 못했습니다: {path}\n"
            "--stats-h5로 학습에 사용한 TRAIN HDF5를 지정하십시오."
        )

    with h5py.File(path, "r") as h5:
        for key in ("feature_mean", "feature_std"):
            if key not in h5:
                raise KeyError(f"stats H5에 {key} dataset이 없습니다: {path}")

        cfg = _read_config_json(h5)
        detected_mode = _detect_feature_mode(h5, cfg)
        feature_mode = detected_mode if requested_mode == "auto" else requested_mode
        if feature_mode not in ("mel129", "stft129"):
            raise ValueError("feature_mode은 auto, mel129, stft129 중 하나여야 합니다.")
        if requested_mode != "auto" and requested_mode != detected_mode:
            raise ValueError(
                f"요청한 feature_mode={requested_mode}와 H5의 mode={detected_mode}가 다릅니다."
            )

        # Both uploaded preprocessors use the same 15 s / 4 kHz / STFT / MFCC settings.
        # The only difference is whether the first 129 values are log-Mel or STFT log-power.
        config = FeatureConfig(
            target_sr=int(cfg.get("target_sr", 4000)),
            record_sec=float(cfg.get("record_sec", 15.0)),
            highpass_hz=float(cfg.get("highpass_hz", 80.0)),
            highpass_order=int(cfg.get("highpass_order", 10)),
            n_fft=int(cfg.get("n_fft", 256)),
            win_length=int(cfg.get("win_length", 256)),
            hop_length=int(cfg.get("hop_length", 64)),
            n_mfcc=int(cfg.get("n_mfcc", 20)),
            spectrogram_n_mels=int(
                cfg.get("spectrogram_n_mels", cfg.get("n_fft", 256) // 2 + 1)
            ),
            mfcc_n_mels=int(cfg.get("mfcc_n_mels", cfg.get("n_mels", 40))),
            delta_width=int(cfg.get("delta_width", 9)),
        )

        feature_mean = np.asarray(h5["feature_mean"], dtype=np.float32).reshape(-1)
        feature_std = np.asarray(h5["feature_std"], dtype=np.float32).reshape(-1)
        clip_raw = float(h5.attrs.get("clip_threshold_train", np.nan))
        clip_threshold = clip_raw if math.isfinite(clip_raw) and clip_raw > 0 else None

        h5_feature_dim = int(h5.attrs.get("feature_dim", len(feature_mean)))
        h5_input_frames = int(h5.attrs.get("input_frames", config.num_frames))

    if config.record_sec != 15.0:
        raise ValueError(f"이 모델은 15초 입력용이어야 하는데 H5 record_sec={config.record_sec}입니다.")
    if config.target_sr != 4000:
        raise ValueError(f"예상 target_sr=4000, H5 target_sr={config.target_sr}")
    if config.num_frames != h5_input_frames:
        raise ValueError(
            f"frame 수 불일치: config={config.num_frames}, H5={h5_input_frames}"
        )
    if h5_feature_dim != 193 or len(feature_mean) != 193 or len(feature_std) != 193:
        raise ValueError(
            "모델 입력은 193차원이어야 합니다. "
            f"H5 feature_dim={h5_feature_dim}, mean={len(feature_mean)}, std={len(feature_std)}"
        )
    if np.any(~np.isfinite(feature_mean)) or np.any(~np.isfinite(feature_std)):
        raise ValueError("feature_mean/std에 NaN 또는 Inf가 있습니다.")
    if np.any(feature_std <= 0):
        raise ValueError("feature_std에는 양수만 있어야 합니다.")

    return TrainingStatistics(
        path=path,
        config=config,
        feature_mean=feature_mean,
        feature_std=feature_std,
        clip_threshold=clip_threshold,
        feature_mode=feature_mode,
    )


def load_fixed_recording(
    path: Path,
    config: FeatureConfig,
    crop_mode: str = "start",
) -> Tuple[np.ndarray, Dict[str, Any]]:
    if crop_mode not in ("start", "center"):
        raise ValueError("crop_mode은 start 또는 center이어야 합니다.")

    waveform, original_sr = sf.read(
        str(path),
        dtype="float32",
        always_2d=True,
    )
    waveform = waveform.mean(axis=1, dtype=np.float32)
    if waveform.size == 0:
        raise ValueError(f"빈 오디오 파일입니다: {path}")
    if not np.isfinite(waveform).all():
        raise ValueError(f"WAV에 NaN 또는 Inf가 있습니다: {path}")

    original_duration = len(waveform) / float(original_sr)
    if original_sr != config.target_sr:
        ratio = Fraction(config.target_sr, int(original_sr)).limit_denominator()
        waveform = resample_poly(
            waveform,
            up=ratio.numerator,
            down=ratio.denominator,
        ).astype(np.float32)

    required = config.required_samples
    resampled_samples = len(waveform)
    crop_start = 0
    pad_samples = 0

    if resampled_samples > required:
        if crop_mode == "center":
            crop_start = (resampled_samples - required) // 2
        waveform = waveform[crop_start:crop_start + required]
        action = "cropped"
    elif resampled_samples < required:
        pad_samples = required - resampled_samples
        waveform = np.pad(waveform, (0, pad_samples), mode="constant")
        action = "zero_padded"
    else:
        action = "unchanged"

    waveform = np.ascontiguousarray(waveform, dtype=np.float32)
    metadata: Dict[str, Any] = {
        "original_sample_rate": int(original_sr),
        "original_duration_sec": float(original_duration),
        "resampled_duration_sec": float(resampled_samples / config.target_sr),
        "processed_duration_sec": float(config.record_sec),
        "crop_start_sec": float(crop_start / config.target_sr),
        "pad_duration_sec": float(pad_samples / config.target_sr),
        "length_action": action,
    }
    return waveform, metadata


def hz_to_mel(hz: np.ndarray | float) -> np.ndarray:
    hz = np.asarray(hz, dtype=np.float64)
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def mel_to_hz(mel: np.ndarray | float) -> np.ndarray:
    mel = np.asarray(mel, dtype=np.float64)
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def make_mel_filterbank(config: FeatureConfig, n_mels: int, slaney_norm: bool) -> np.ndarray:
    n_freq = config.n_fft // 2 + 1
    mel_points = np.linspace(
        hz_to_mel(0.0),
        hz_to_mel(config.target_sr / 2.0),
        n_mels + 2,
    )
    hz_points = mel_to_hz(mel_points)
    bins = np.floor((config.n_fft + 1) * hz_points / config.target_sr).astype(int)
    bins = np.clip(bins, 0, n_freq - 1)

    filters = np.zeros((n_mels, n_freq), dtype=np.float32)
    for index in range(1, n_mels + 1):
        left = bins[index - 1]
        center = bins[index]
        right = bins[index + 1]

        if center <= left:
            center = min(left + 1, n_freq - 1)
        if right <= center:
            right = min(center + 1, n_freq)

        if center > left:
            filters[index - 1, left:center] = (
                np.arange(left, center) - left
            ) / float(center - left)
        if right > center:
            filters[index - 1, center:right] = (
                right - np.arange(center, right)
            ) / float(right - center)

    if slaney_norm:
        enorm = 2.0 / np.maximum(hz_points[2:n_mels + 2] - hz_points[:n_mels], 1e-12)
        filters *= enorm[:, None].astype(np.float32)
    return filters


def make_dct_basis(n_mfcc: int, n_mels: int) -> np.ndarray:
    n = np.arange(n_mels, dtype=np.float64)
    k = np.arange(n_mfcc, dtype=np.float64)[:, None]
    basis = np.cos(np.pi / n_mels * (n + 0.5) * k)
    basis[0] *= np.sqrt(1.0 / n_mels)
    if n_mfcc > 1:
        basis[1:] *= np.sqrt(2.0 / n_mels)
    return basis.astype(np.float32)


class ExactFeatureExtractor:
    """Exact inference implementation of the uploaded TRAIN preprocessing."""

    def __init__(
        self,
        statistics: TrainingStatistics,
        device: str,
    ) -> None:
        self.statistics = statistics
        self.config = statistics.config
        self.feature_mode = statistics.feature_mode
        self.device = torch.device(device)

        self.sos = butter(
            self.config.highpass_order,
            self.config.highpass_hz,
            btype="highpass",
            fs=self.config.target_sr,
            output="sos",
        )
        self.window = torch.hann_window(
            self.config.win_length,
            periodic=True,
            dtype=torch.float32,
            device=self.device,
        )

        # Uploaded mel129 preprocessing uses Slaney area normalization for both banks.
        # Uploaded stft129 preprocessing uses the unnormalized 40-bin MFCC bank.
        slaney = self.feature_mode == "mel129"
        if self.feature_mode == "mel129":
            self.spectrogram_mel_filterbank = torch.as_tensor(
                make_mel_filterbank(
                    self.config,
                    self.config.spectrogram_n_mels,
                    slaney_norm=True,
                ),
                dtype=torch.float32,
                device=self.device,
            )
        else:
            self.spectrogram_mel_filterbank = None

        self.mfcc_mel_filterbank = torch.as_tensor(
            make_mel_filterbank(
                self.config,
                self.config.mfcc_n_mels,
                slaney_norm=slaney,
            ),
            dtype=torch.float32,
            device=self.device,
        )
        self.dct_basis = torch.as_tensor(
            make_dct_basis(self.config.n_mfcc, self.config.mfcc_n_mels),
            dtype=torch.float32,
            device=self.device,
        )

        if self.config.delta_width < 3 or self.config.delta_width % 2 == 0:
            raise ValueError("delta_width는 3 이상의 홀수여야 합니다.")
        half = self.config.delta_width // 2
        offsets = torch.arange(
            -half,
            half + 1,
            dtype=torch.float32,
            device=self.device,
        )
        denominator = 2.0 * sum(v * v for v in range(1, half + 1))
        self.delta_kernel = (offsets / denominator).view(1, 1, -1)

        frequencies = torch.fft.rfftfreq(
            self.config.n_fft,
            d=1.0 / self.config.target_sr,
        ).cpu().numpy().astype(np.float32)
        self.frequencies_hz = frequencies
        self.band_masks = []
        for low, high in (
            (0.0, 250.0),
            (250.0, 500.0),
            (500.0, 1000.0),
            (0.0, 2000.0),
        ):
            if high >= self.config.target_sr / 2:
                mask = (frequencies >= low) & (frequencies <= high)
            else:
                mask = (frequencies >= low) & (frequencies < high)
            self.band_masks.append(torch.as_tensor(mask, device=self.device))

        self.feature_mean = torch.as_tensor(
            statistics.feature_mean,
            dtype=torch.float32,
            device=self.device,
        ).view(1, 1, -1)
        self.feature_std = torch.as_tensor(
            statistics.feature_std,
            dtype=torch.float32,
            device=self.device,
        ).view(1, 1, -1)

    def delta(self, values: torch.Tensor) -> torch.Tensor:
        channels = values.shape[2]
        half = self.config.delta_width // 2
        channel_first = values.transpose(1, 2)
        padded = F.pad(channel_first, (half, half), mode="replicate")
        kernel = self.delta_kernel.repeat(channels, 1, 1)
        result = F.conv1d(padded, kernel, groups=channels)
        return result.transpose(1, 2)

    def filter_and_clip(self, waveform: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
        x = np.asarray(waveform, dtype=np.float32)
        x = x - np.mean(x, dtype=np.float64)
        x = sosfiltfilt(self.sos, x).astype(np.float32)
        x = np.ascontiguousarray(x, dtype=np.float32)

        peak_before_clip = float(np.max(np.abs(x)))
        threshold = self.statistics.clip_threshold
        if threshold is None:
            clipping_ratio = 0.0
        else:
            clipping_ratio = float(np.mean(np.abs(x) > threshold))
            x = np.clip(x, -threshold, threshold).astype(np.float32)
        return x, {
            "peak_before_clip": peak_before_clip,
            "clipping_ratio": clipping_ratio,
        }

    @torch.inference_mode()
    def transform_raw(self, waveform: np.ndarray) -> torch.Tensor:
        x = torch.as_tensor(
            waveform[None, :],
            dtype=torch.float32,
            device=self.device,
        )
        stft = torch.stft(
            x,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            win_length=self.config.win_length,
            window=self.window,
            center=True,
            pad_mode="reflect",
            return_complex=True,
        )
        power = stft.abs().square()

        if self.feature_mode == "mel129":
            spectrogram_mel_power = torch.einsum(
                "mf,bft->bmt",
                self.spectrogram_mel_filterbank,
                power,
            ).clamp_min(1e-10)
            spectrogram = (10.0 * torch.log10(spectrogram_mel_power)).transpose(1, 2)
        else:
            spectrogram = (10.0 * torch.log10(power.clamp_min(1e-10))).transpose(1, 2)

        mfcc_mel_power = torch.einsum(
            "mf,bft->bmt",
            self.mfcc_mel_filterbank,
            power,
        ).clamp_min(1e-10)
        log_mel_for_mfcc = torch.log(mfcc_mel_power)
        mfcc = torch.einsum("km,bmt->btk", self.dct_basis, log_mel_for_mfcc)
        mfcc_delta = self.delta(mfcc)
        mfcc_delta2 = self.delta(mfcc_delta)
        mfcc_group = torch.cat([mfcc, mfcc_delta, mfcc_delta2], dim=2)

        band_energy = torch.stack(
            [
                10.0
                * torch.log10(
                    power[:, mask, :].sum(dim=1).clamp_min(1e-10)
                )
                for mask in self.band_masks
            ],
            dim=2,
        )

        features = torch.cat([spectrogram, mfcc_group, band_energy], dim=2)
        expected = (1, self.config.num_frames, 193)
        if tuple(features.shape) != expected:
            raise RuntimeError(f"feature shape={tuple(features.shape)}, expected={expected}")
        return features

    @torch.inference_mode()
    def preprocess_file(
        self,
        wav_path: Path,
        crop_mode: str = "start",
    ) -> Dict[str, Any]:
        fixed_waveform, length_info = load_fixed_recording(
            wav_path,
            self.config,
            crop_mode=crop_mode,
        )
        processed_waveform, signal_info = self.filter_and_clip(fixed_waveform)
        raw_features = self.transform_raw(processed_waveform)
        standardized = (raw_features - self.feature_mean) / self.feature_std
        standardized = torch.clamp(standardized, -12.0, 12.0)

        # Training preprocessing stores features as float16 in HDF5.
        # Quantize once here, then convert back to float32 for a stable model input.
        # The numerical values are therefore the same values read from HDF5.
        standardized_f16 = standardized.to(torch.float16)
        model_features = standardized_f16.to(torch.float32)

        result: Dict[str, Any] = {
            "fixed_waveform": fixed_waveform,
            "processed_waveform": processed_waveform,
            "raw_features": raw_features.squeeze(0),
            "standardized_features": model_features.squeeze(0),
            "standardized_features_f16": standardized_f16.squeeze(0),
            "spectral_feature": model_features[0, :, :129].transpose(0, 1),
            "feature_times": np.arange(self.config.num_frames, dtype=np.float64)
            * self.config.hop_length
            / self.config.target_sr,
        }
        result.update(length_info)
        result.update(signal_info)
        return result

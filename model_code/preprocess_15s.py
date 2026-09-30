#!/usr/bin/env python3
"""
HF_Lung_V1 raw WAV/TXT -> 15-second joint I/E HDF5.

Default input
-------------
/home/tta/BreathOn/HF_Lung_V1/
  train/...
  test/...

Default output
--------------
/home/tta/Woo_code/data/HF_Lung_V1_pre_joint15/
  HF_Lung_V1_train_15s_logstd.h5
  HF_Lung_V1_test_15s_logstd.h5

Processing
----------
- Use one full 15-second sample per recording. No 10-second crop and no sliding window.
- Resample to 4,000 Hz.
- Remove DC and apply a 10th-order 80-Hz Butterworth high-pass filter.
- Estimate one symmetric robust clipping threshold from TRAIN waveforms only.
- STFT: n_fft=256, win_length=256, hop_length=64, Hann, center=True.
- Create 193-dimensional features:
    129 log-power spectrogram bins
     20 MFCC
     20 MFCC delta
     20 MFCC delta-delta
      4 log band energies
- Estimate per-feature mean/std from TRAIN frames only.
- Apply the same TRAIN statistics to TRAIN and TEST.
- Store standardized features as float16.
- Create joint I/E binary labels:
    frame_target: [N, 938, 2]
    target:       [N, 469, 2]

This is a new robust preprocessing experiment. It is not a bit-for-bit
reproduction of the paper's private feature implementation.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
import sys
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable, Sequence

import h5py
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from scipy.signal import butter, resample_poly, sosfiltfilt
from tqdm import tqdm


LOGGER = logging.getLogger("hf_lung_15s_logstd")

TARGET_ORDER = ("I", "E")
TARGET_INDEX = {"I": 0, "E": 1}

LABEL_ALIASES = {
    "I": "I",
    "INHALE": "I",
    "INHALATION": "I",
    "INSPIRATION": "I",
    "INSPIRATORY": "I",
    "E": "E",
    "EXHALE": "E",
    "EXHALATION": "E",
    "EXPIRATION": "E",
    "EXPIRATORY": "E",
    "W": "W",
    "WHEEZE": "W",
    "WHEEZES": "W",
    "S": "S",
    "STRIDOR": "S",
    "R": "R",
    "RHONCHI": "R",
    "RHONCHUS": "R",
    "D": "D",
    "DAS": "D",
    "CRACKLE": "D",
    "CRACKLES": "D",
    "C": "C",
    "CAS": "C",
}


@dataclass(frozen=True)
class Event:
    label: str
    start_sec: float
    end_sec: float


@dataclass(frozen=True)
class Config:
    raw_root: str
    out_root: str
    target_sr: int = 4000
    record_sec: float = 15.0
    highpass_hz: float = 80.0
    highpass_order: int = 10
    n_fft: int = 256
    win_length: int = 256
    hop_length: int = 64
    n_mfcc: int = 20
    n_mels: int = 40
    delta_width: int = 9
    clip_percentile: float = 99.99
    clip_reservoir_size: int = 2_000_000
    batch_size: int = 32
    device: str = "cuda"
    compression: str = "lzf"
    allow_missing_labels: bool = False
    pad_short: bool = False
    subject_regex: str | None = None
    random_seed: int = 2026


class ReservoirSampler:
    def __init__(self, capacity: int, seed: int) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = int(capacity)
        self.rng = np.random.default_rng(seed)
        self.data = np.empty(self.capacity, dtype=np.float32)
        self.size = 0
        self.seen = 0

    def update(
        self,
        values: np.ndarray,
        max_values_per_recording: int = 4096,
    ) -> None:
        x = np.abs(np.asarray(values, dtype=np.float32).reshape(-1))
        if x.size > max_values_per_recording:
            indices = self.rng.choice(
                x.size,
                size=max_values_per_recording,
                replace=False,
            )
            x = x[indices]

        for value in x:
            self.seen += 1
            if self.size < self.capacity:
                self.data[self.size] = value
                self.size += 1
            else:
                index = int(self.rng.integers(0, self.seen))
                if index < self.capacity:
                    self.data[index] = value

    def percentile(self, q: float) -> float:
        if self.size == 0:
            raise RuntimeError("clipping reservoir is empty")
        return float(np.percentile(self.data[: self.size], q))


class VectorStreamingStats:
    """Streaming per-feature mean/std for arrays shaped [B,T,D]."""

    def __init__(self, feature_dim: int) -> None:
        self.feature_dim = int(feature_dim)
        self.count = 0
        self.sum = np.zeros(self.feature_dim, dtype=np.float64)
        self.sum_sq = np.zeros(self.feature_dim, dtype=np.float64)

    def update(self, values: np.ndarray) -> None:
        x = np.asarray(values, dtype=np.float64)
        if x.ndim != 3 or x.shape[-1] != self.feature_dim:
            raise ValueError(
                f"Expected [B,T,{self.feature_dim}], got {x.shape}"
            )
        flattened = x.reshape(-1, self.feature_dim)
        self.count += flattened.shape[0]
        self.sum += flattened.sum(axis=0)
        self.sum_sq += np.square(flattened).sum(axis=0)

    def finalize(self) -> tuple[np.ndarray, np.ndarray]:
        if self.count < 2:
            raise RuntimeError("not enough values for feature statistics")

        mean = self.sum / self.count
        variance = self.sum_sq / self.count - np.square(mean)
        variance = np.maximum(variance, 1e-8)
        std = np.sqrt(variance)

        return mean.astype(np.float32), std.astype(np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess full 15-second HF_Lung recordings."
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("/home/tta/BreathOn/HF_Lung_V1"),
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path(
            "/home/tta/Woo_code/data/HF_Lung_V1_pre_joint15"
        ),
    )
    parser.add_argument("--target-sr", type=int, default=4000)
    parser.add_argument("--record-sec", type=float, default=15.0)
    parser.add_argument("--highpass-hz", type=float, default=80.0)
    parser.add_argument("--highpass-order", type=int, default=10)
    parser.add_argument("--n-fft", type=int, default=256)
    parser.add_argument("--win-length", type=int, default=256)
    parser.add_argument("--hop-length", type=int, default=64)
    parser.add_argument("--n-mfcc", type=int, default=20)
    parser.add_argument("--n-mels", type=int, default=40)
    parser.add_argument("--delta-width", type=int, default=9)
    parser.add_argument("--clip-percentile", type=float, default=99.99)
    parser.add_argument(
        "--no-clip",
        action="store_true",
        help="Disable TRAIN-derived robust waveform clipping.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--compression",
        choices=["lzf", "gzip", "none"],
        default="lzf",
    )
    parser.add_argument(
        "--allow-missing-labels",
        action="store_true",
    )
    parser.add_argument(
        "--pad-short",
        action="store_true",
        help="Zero-pad recordings shorter than 15 seconds.",
    )
    parser.add_argument(
        "--subject-regex",
        default=None,
        help=(
            "Optional regex applied to the split-relative WAV path. "
            "The first capture group is stored as subject_id."
        ),
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def configure_logging(out_root: Path) -> None:
    out_root.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(
                out_root / "preprocess_15s_logstd.log",
                mode="w",
                encoding="utf-8",
            ),
        ],
    )


def resolve_split_dir(raw_root: Path, split_name: str) -> Path:
    direct = raw_root / split_name
    if direct.is_dir():
        return direct

    matches = [
        path
        for path in raw_root.iterdir()
        if path.is_dir() and path.name.lower() == split_name.lower()
    ]
    if len(matches) == 1:
        return matches[0]

    raise FileNotFoundError(
        f"Could not resolve split {split_name!r} under {raw_root}"
    )


def find_wavs(split_dir: Path) -> list[Path]:
    paths = sorted(
        path
        for pattern in ("*.wav", "*.WAV")
        for path in split_dir.rglob(pattern)
    )
    return list(dict.fromkeys(paths))


def normalized_label_stem(path: Path) -> str:
    stem = path.stem.lower()
    for suffix in (
        "_labels",
        "_label",
        "-labels",
        "-label",
        ".labels",
        ".label",
    ):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
    return stem


def build_label_index(split_dir: Path) -> dict[str, list[Path]]:
    index: dict[str, list[Path]] = {}
    for path in sorted(split_dir.rglob("*.txt")):
        index.setdefault(normalized_label_stem(path), []).append(path)
    return index


def choose_label_path(
    wav_path: Path,
    label_index: dict[str, list[Path]],
) -> Path | None:
    candidates = label_index.get(wav_path.stem.lower(), [])
    if not candidates:
        candidates = label_index.get(normalized_label_stem(wav_path), [])
    if not candidates:
        return None

    candidates = sorted(
        candidates,
        key=lambda path: (
            len(
                set(wav_path.parents).symmetric_difference(
                    set(path.parents)
                )
            ),
            len(str(path)),
        ),
    )

    if len(candidates) > 1:
        LOGGER.warning(
            "Multiple label candidates for %s; using %s",
            wav_path,
            candidates[0],
        )
    return candidates[0]


def normalize_label_name(token: str) -> str | None:
    token = re.sub(r"[^A-Za-z]", "", token).upper()
    return LABEL_ALIASES.get(token)


def parse_time(token: str) -> float | None:
    token = token.strip()
    try:
        parts = token.split(":")
        if len(parts) == 1:
            value = float(parts[0])
        elif len(parts) == 2:
            minutes = float(parts[0])
            seconds = float(parts[1])
            if not 0 <= seconds < 60:
                return None
            value = 60.0 * minutes + seconds
        elif len(parts) == 3:
            hours = float(parts[0])
            minutes = float(parts[1])
            seconds = float(parts[2])
            if not 0 <= minutes < 60:
                return None
            if not 0 <= seconds < 60:
                return None
            value = 3600.0 * hours + 60.0 * minutes + seconds
        else:
            return None
    except ValueError:
        return None

    return value if math.isfinite(value) else None


def parse_label_file(path: Path) -> list[Event]:
    events: list[Event] = []

    with path.open(
        "r",
        encoding="utf-8-sig",
        errors="replace",
    ) as file:
        for line_number, raw_line in enumerate(file, start=1):
            line = raw_line.split("#", 1)[0].strip()
            if not line:
                continue

            tokens = [
                token
                for token in re.split(r"[\s,;]+", line)
                if token
            ]
            if len(tokens) < 3:
                LOGGER.warning(
                    "Malformed label row %s:%d: %r",
                    path,
                    line_number,
                    raw_line.rstrip(),
                )
                continue

            first_label = normalize_label_name(tokens[0])
            last_label = normalize_label_name(tokens[-1])

            label: str | None = None
            start: float | None = None
            end: float | None = None

            if first_label is not None:
                label = first_label
                start = parse_time(tokens[1])
                end = parse_time(tokens[2])
            elif last_label is not None:
                label = last_label
                start = parse_time(tokens[0])
                end = parse_time(tokens[1])

            if label is None or start is None or end is None:
                LOGGER.warning(
                    "Could not parse label row %s:%d: %r",
                    path,
                    line_number,
                    raw_line.rstrip(),
                )
                continue

            if start < 0 or end <= start:
                LOGGER.warning(
                    "Invalid label interval %s:%d: %s %.6f %.6f",
                    path,
                    line_number,
                    label,
                    start,
                    end,
                )
                continue

            events.append(Event(label, float(start), float(end)))

    events.sort(key=lambda event: (event.start_sec, event.end_sec))
    return events


def infer_subject_id(
    wav_path: Path,
    split_dir: Path,
    subject_regex: str | None,
) -> str:
    relative = wav_path.relative_to(split_dir)
    text = relative.as_posix()

    if subject_regex:
        match = re.search(subject_regex, text)
        if match is None:
            raise ValueError(
                f"subject regex {subject_regex!r} did not match {text!r}"
            )
        return match.group(1) if match.groups() else match.group(0)

    if len(relative.parts) > 1:
        return relative.parts[0]

    return wav_path.stem


def load_recording(
    path: Path,
    target_sr: int,
    record_sec: float,
    pad_short: bool,
) -> tuple[np.ndarray, int, float]:
    waveform, original_sr = sf.read(
        path,
        dtype="float32",
        always_2d=True,
    )
    waveform = waveform.mean(axis=1, dtype=np.float32)

    if not np.isfinite(waveform).all():
        raise ValueError(f"NaN or Inf found in {path}")

    original_duration = len(waveform) / float(original_sr)

    if original_sr != target_sr:
        ratio = Fraction(target_sr, original_sr).limit_denominator()
        waveform = resample_poly(
            waveform,
            up=ratio.numerator,
            down=ratio.denominator,
        ).astype(np.float32)

    required_samples = int(round(record_sec * target_sr))
    if len(waveform) < required_samples:
        if not pad_short:
            raise ValueError(
                f"Recording shorter than {record_sec}s: "
                f"{len(waveform) / target_sr:.3f}s"
            )
        waveform = np.pad(
            waveform,
            (0, required_samples - len(waveform)),
        )
    else:
        waveform = waveform[:required_samples]

    return (
        np.ascontiguousarray(waveform, dtype=np.float32),
        int(original_sr),
        float(original_duration),
    )


def make_highpass_sos(config: Config) -> np.ndarray:
    return butter(
        config.highpass_order,
        config.highpass_hz,
        btype="highpass",
        fs=config.target_sr,
        output="sos",
    )


def filter_recording(
    waveform: np.ndarray,
    sos: np.ndarray,
) -> np.ndarray:
    waveform = np.asarray(waveform, dtype=np.float32)
    waveform = waveform - np.mean(waveform, dtype=np.float64)
    filtered = sosfiltfilt(sos, waveform)
    return np.ascontiguousarray(filtered, dtype=np.float32)


def apply_clip(
    waveform: np.ndarray,
    threshold: float | None,
) -> tuple[np.ndarray, float, float]:
    peak_before_clip = float(np.max(np.abs(waveform)))

    if threshold is None:
        return waveform, peak_before_clip, 0.0

    clipping_ratio = float(np.mean(np.abs(waveform) > threshold))
    waveform = np.clip(
        waveform,
        -threshold,
        threshold,
    ).astype(np.float32)

    return waveform, peak_before_clip, clipping_ratio


def hz_to_mel(hz: np.ndarray | float) -> np.ndarray:
    hz = np.asarray(hz, dtype=np.float64)
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def mel_to_hz(mel: np.ndarray | float) -> np.ndarray:
    mel = np.asarray(mel, dtype=np.float64)
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def make_mel_filterbank(config: Config) -> np.ndarray:
    n_freq = config.n_fft // 2 + 1
    mel_points = np.linspace(
        hz_to_mel(0.0),
        hz_to_mel(config.target_sr / 2.0),
        config.n_mels + 2,
    )
    hz_points = mel_to_hz(mel_points)
    bins = np.floor(
        (config.n_fft + 1) * hz_points / config.target_sr
    ).astype(int)
    bins = np.clip(bins, 0, n_freq - 1)

    filters = np.zeros(
        (config.n_mels, n_freq),
        dtype=np.float32,
    )

    for index in range(1, config.n_mels + 1):
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

    return filters


def make_dct_basis(n_mfcc: int, n_mels: int) -> np.ndarray:
    n = np.arange(n_mels, dtype=np.float64)
    k = np.arange(n_mfcc, dtype=np.float64)[:, None]
    basis = np.cos(
        np.pi / n_mels * (n + 0.5) * k
    )
    basis[0] *= np.sqrt(1.0 / n_mels)
    if n_mfcc > 1:
        basis[1:] *= np.sqrt(2.0 / n_mels)
    return basis.astype(np.float32)


class FeatureExtractor:
    def __init__(self, config: Config) -> None:
        self.config = config

        requested_device = config.device
        if (
            requested_device.startswith("cuda")
            and not torch.cuda.is_available()
        ):
            LOGGER.warning(
                "CUDA requested but unavailable; using CPU."
            )
            requested_device = "cpu"

        self.device = torch.device(requested_device)
        self.window = torch.hann_window(
            config.win_length,
            periodic=True,
            dtype=torch.float32,
            device=self.device,
        )
        self.mel_filterbank = torch.as_tensor(
            make_mel_filterbank(config),
            dtype=torch.float32,
            device=self.device,
        )
        self.dct_basis = torch.as_tensor(
            make_dct_basis(config.n_mfcc, config.n_mels),
            dtype=torch.float32,
            device=self.device,
        )

        if config.delta_width < 3 or config.delta_width % 2 == 0:
            raise ValueError(
                "delta_width must be an odd integer >= 3"
            )

        half = config.delta_width // 2
        offsets = torch.arange(
            -half,
            half + 1,
            dtype=torch.float32,
            device=self.device,
        )
        denominator = 2.0 * sum(
            value * value
            for value in range(1, half + 1)
        )
        self.delta_kernel = (
            offsets / denominator
        ).view(1, 1, -1)

        self.frequencies_hz = torch.fft.rfftfreq(
            config.n_fft,
            d=1.0 / config.target_sr,
        ).cpu().numpy().astype(np.float32)

        self.band_masks: list[torch.Tensor] = []
        for low, high in (
            (0.0, 250.0),
            (250.0, 500.0),
            (500.0, 1000.0),
            (0.0, 2000.0),
        ):
            if high >= config.target_sr / 2:
                mask = (
                    (self.frequencies_hz >= low)
                    & (self.frequencies_hz <= high)
                )
            else:
                mask = (
                    (self.frequencies_hz >= low)
                    & (self.frequencies_hz < high)
                )
            self.band_masks.append(
                torch.as_tensor(mask, device=self.device)
            )

        required_samples = int(
            round(config.record_sec * config.target_sr)
        )
        self.num_frames = (
            required_samples // config.hop_length + 1
        )
        self.feature_dim = (
            config.n_fft // 2 + 1
            + 3 * config.n_mfcc
            + 4
        )

        if self.num_frames != 938:
            LOGGER.warning(
                "Current configuration produces %d frames, not 938.",
                self.num_frames,
            )
        if self.feature_dim != 193:
            LOGGER.warning(
                "Current configuration produces %d features, not 193.",
                self.feature_dim,
            )

    def delta(self, values: torch.Tensor) -> torch.Tensor:
        channels = values.shape[2]
        half = self.config.delta_width // 2
        channel_first = values.transpose(1, 2)
        padded = F.pad(
            channel_first,
            (half, half),
            mode="replicate",
        )
        kernel = self.delta_kernel.repeat(
            channels,
            1,
            1,
        )
        result = F.conv1d(
            padded,
            kernel,
            groups=channels,
        )
        return result.transpose(1, 2)

    @torch.inference_mode()
    def transform_raw(
        self,
        waveform_batch: np.ndarray,
    ) -> np.ndarray:
        waveform = torch.as_tensor(
            waveform_batch,
            dtype=torch.float32,
            device=self.device,
        )

        stft = torch.stft(
            waveform,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            win_length=self.config.win_length,
            window=self.window,
            center=True,
            pad_mode="reflect",
            return_complex=True,
        )
        power = stft.abs().square()

        # 129-dimensional log-power spectrogram in dB.
        log_power_db = 10.0 * torch.log10(
            power.clamp_min(1e-10)
        )
        spectrogram = log_power_db.transpose(1, 2)

        # 20 MFCC + 20 delta + 20 delta-delta.
        mel_power = torch.einsum(
            "mf,bft->bmt",
            self.mel_filterbank,
            power,
        ).clamp_min(1e-10)
        log_mel = torch.log(mel_power)
        mfcc = torch.einsum(
            "km,bmt->btk",
            self.dct_basis,
            log_mel,
        )
        mfcc_delta = self.delta(mfcc)
        mfcc_delta2 = self.delta(mfcc_delta)
        mfcc_group = torch.cat(
            [mfcc, mfcc_delta, mfcc_delta2],
            dim=2,
        )

        # Four log band energies in dB.
        band_energy = torch.stack(
            [
                10.0
                * torch.log10(
                    power[:, mask, :]
                    .sum(dim=1)
                    .clamp_min(1e-10)
                )
                for mask in self.band_masks
            ],
            dim=2,
        )

        features = torch.cat(
            [spectrogram, mfcc_group, band_energy],
            dim=2,
        )

        if features.shape[1:] != (
            self.num_frames,
            self.feature_dim,
        ):
            raise RuntimeError(
                f"Unexpected feature shape {tuple(features.shape)}"
            )

        return features.cpu().numpy().astype(np.float32)


def build_labels(
    events: Sequence[Event],
    config: Config,
    num_frames: int,
) -> tuple[np.ndarray, np.ndarray]:
    # With center=True, frame index j is centered at j * hop / sr.
    frame_times = (
        np.arange(num_frames, dtype=np.float64)
        * config.hop_length
        / config.target_sr
    )

    frame_target = np.zeros(
        (num_frames, 2),
        dtype=np.uint8,
    )

    for event in events:
        if event.label not in TARGET_INDEX:
            continue

        start = max(0.0, event.start_sec)
        end = min(config.record_sec, event.end_sec)
        if end <= start:
            continue

        positive = (
            (frame_times >= start)
            & (frame_times < end)
        )
        frame_target[
            positive,
            TARGET_INDEX[event.label],
        ] = 1

    if num_frames % 2:
        padded = np.pad(
            frame_target,
            ((0, 1), (0, 0)),
            mode="constant",
        )
    else:
        padded = frame_target

    target = padded.reshape(
        -1,
        2,
        2,
    ).max(axis=1)

    return frame_target, target.astype(np.uint8)


def inspect_matching(
    split_dir: Path,
) -> dict[str, Any]:
    wavs = find_wavs(split_dir)
    label_index = build_label_index(split_dir)

    matched: list[dict[str, str]] = []
    missing: list[str] = []

    for wav_path in wavs:
        label_path = choose_label_path(
            wav_path,
            label_index,
        )
        if label_path is None:
            missing.append(str(wav_path))
        else:
            matched.append(
                {
                    "wav": str(wav_path),
                    "label": str(label_path),
                }
            )

    return {
        "split_dir": str(split_dir),
        "wav_count": len(wavs),
        "matched_count": len(matched),
        "missing_label_count": len(missing),
        "matched_examples": matched[:5],
        "missing_examples": missing[:20],
    }


def estimate_clip_threshold(
    train_dir: Path,
    config: Config,
    sos: np.ndarray,
) -> tuple[float, list[dict[str, str]]]:
    sampler = ReservoirSampler(
        capacity=config.clip_reservoir_size,
        seed=config.random_seed,
    )
    failures: list[dict[str, str]] = []
    label_index = build_label_index(train_dir)

    for wav_path in tqdm(
        find_wavs(train_dir),
        desc="Pass 1/3: TRAIN clipping threshold",
    ):
        label_path = choose_label_path(
            wav_path,
            label_index,
        )
        if (
            label_path is None
            and not config.allow_missing_labels
        ):
            continue

        try:
            waveform, _, _ = load_recording(
                wav_path,
                config.target_sr,
                config.record_sec,
                config.pad_short,
            )
            filtered = filter_recording(
                waveform,
                sos,
            )
            sampler.update(filtered)
        except Exception as exc:
            failures.append(
                {
                    "file": str(wav_path),
                    "error": repr(exc),
                }
            )
            LOGGER.exception(
                "Clipping pass failed: %s",
                wav_path,
            )

    threshold = sampler.percentile(
        config.clip_percentile
    )
    if not math.isfinite(threshold) or threshold <= 0:
        raise RuntimeError(
            f"Invalid clipping threshold: {threshold}"
        )

    return threshold, failures


def collect_train_feature_stats(
    train_dir: Path,
    config: Config,
    sos: np.ndarray,
    clip_threshold: float | None,
    extractor: FeatureExtractor,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    stats = VectorStreamingStats(
        extractor.feature_dim
    )
    failures: list[dict[str, str]] = []
    label_index = build_label_index(train_dir)

    waveform_batch: list[np.ndarray] = []

    def flush() -> None:
        if not waveform_batch:
            return
        features = extractor.transform_raw(
            np.stack(waveform_batch, axis=0)
        )
        stats.update(features)
        waveform_batch.clear()

    for wav_path in tqdm(
        find_wavs(train_dir),
        desc="Pass 2/3: TRAIN feature statistics",
    ):
        label_path = choose_label_path(
            wav_path,
            label_index,
        )
        if (
            label_path is None
            and not config.allow_missing_labels
        ):
            continue

        try:
            waveform, _, _ = load_recording(
                wav_path,
                config.target_sr,
                config.record_sec,
                config.pad_short,
            )
            waveform = filter_recording(
                waveform,
                sos,
            )
            waveform, _, _ = apply_clip(
                waveform,
                clip_threshold,
            )
            waveform_batch.append(waveform)

            if len(waveform_batch) >= config.batch_size:
                flush()
        except Exception as exc:
            failures.append(
                {
                    "file": str(wav_path),
                    "error": repr(exc),
                }
            )
            LOGGER.exception(
                "Feature-stat pass failed: %s",
                wav_path,
            )

    flush()
    mean, std = stats.finalize()

    report = {
        "frame_count": int(stats.count),
        "feature_dim": int(extractor.feature_dim),
        "mean_min": float(mean.min()),
        "mean_max": float(mean.max()),
        "std_min": float(std.min()),
        "std_max": float(std.max()),
        "failure_count": len(failures),
        "failures": failures[:100],
    }
    return mean, std, report


def compression_name(name: str) -> str | None:
    return None if name == "none" else name


def create_datasets(
    h5: h5py.File,
    max_samples: int,
    extractor: FeatureExtractor,
    compression: str | None,
) -> dict[str, h5py.Dataset]:
    string_dtype = h5py.string_dtype(
        encoding="utf-8"
    )
    output_frames = (
        extractor.num_frames + 1
    ) // 2

    return {
        "features": h5.create_dataset(
            "features",
            shape=(
                max_samples,
                extractor.num_frames,
                extractor.feature_dim,
            ),
            maxshape=(
                None,
                extractor.num_frames,
                extractor.feature_dim,
            ),
            dtype=np.float16,
            chunks=(
                1,
                extractor.num_frames,
                extractor.feature_dim,
            ),
            compression=compression,
            shuffle=bool(compression),
        ),
        "frame_target": h5.create_dataset(
            "frame_target",
            shape=(
                max_samples,
                extractor.num_frames,
                2,
            ),
            maxshape=(
                None,
                extractor.num_frames,
                2,
            ),
            dtype=np.uint8,
            chunks=(
                8,
                extractor.num_frames,
                2,
            ),
            compression=compression,
            shuffle=bool(compression),
        ),
        "target": h5.create_dataset(
            "target",
            shape=(
                max_samples,
                output_frames,
                2,
            ),
            maxshape=(
                None,
                output_frames,
                2,
            ),
            dtype=np.uint8,
            chunks=(
                8,
                output_frames,
                2,
            ),
            compression=compression,
            shuffle=bool(compression),
        ),
        "label_presence": h5.create_dataset(
            "label_presence",
            shape=(max_samples, 2),
            maxshape=(None, 2),
            dtype=np.uint8,
        ),
        "source_file": h5.create_dataset(
            "source_file",
            shape=(max_samples,),
            maxshape=(None,),
            dtype=string_dtype,
        ),
        "label_file": h5.create_dataset(
            "label_file",
            shape=(max_samples,),
            maxshape=(None,),
            dtype=string_dtype,
        ),
        "subject_id": h5.create_dataset(
            "subject_id",
            shape=(max_samples,),
            maxshape=(None,),
            dtype=string_dtype,
        ),
        "original_sample_rate": h5.create_dataset(
            "original_sample_rate",
            shape=(max_samples,),
            maxshape=(None,),
            dtype=np.int32,
        ),
        "recording_duration_sec": h5.create_dataset(
            "recording_duration_sec",
            shape=(max_samples,),
            maxshape=(None,),
            dtype=np.float32,
        ),
        "peak_before_clip": h5.create_dataset(
            "peak_before_clip",
            shape=(max_samples,),
            maxshape=(None,),
            dtype=np.float32,
        ),
        "clipping_ratio": h5.create_dataset(
            "clipping_ratio",
            shape=(max_samples,),
            maxshape=(None,),
            dtype=np.float32,
        ),
    }


def resize_datasets(
    datasets: dict[str, h5py.Dataset],
    size: int,
) -> None:
    for dataset in datasets.values():
        dataset.resize(size, axis=0)


def write_split(
    split_name: str,
    split_dir: Path,
    output_path: Path,
    config: Config,
    sos: np.ndarray,
    clip_threshold: float | None,
    extractor: FeatureExtractor,
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    overwrite: bool,
) -> dict[str, Any]:
    if output_path.exists():
        if not overwrite:
            raise FileExistsError(
                f"{output_path} exists. Pass --overwrite."
            )
        output_path.unlink()

    wavs = find_wavs(split_dir)
    label_index = build_label_index(split_dir)
    compression = compression_name(
        config.compression
    )

    failures: list[dict[str, str]] = []
    missing_labels: list[str] = []
    written = 0

    waveform_batch: list[np.ndarray] = []
    metadata_batch: list[dict[str, Any]] = []

    with h5py.File(output_path, "w") as h5:
        datasets = create_datasets(
            h5,
            max_samples=len(wavs),
            extractor=extractor,
            compression=compression,
        )

        h5.create_dataset(
            "frequencies_hz",
            data=extractor.frequencies_hz,
        )
        h5.create_dataset(
            "feature_mean",
            data=feature_mean.astype(np.float32),
        )
        h5.create_dataset(
            "feature_std",
            data=feature_std.astype(np.float32),
        )

        h5.attrs["split"] = split_name
        h5.attrs["sample_count"] = 0
        h5.attrs["target_order"] = json.dumps(
            TARGET_ORDER
        )
        h5.attrs["target_index_I"] = 0
        h5.attrs["target_index_E"] = 1
        h5.attrs["input_frames"] = extractor.num_frames
        h5.attrs["feature_dim"] = extractor.feature_dim
        h5.attrs["output_segments"] = (
            extractor.num_frames + 1
        ) // 2
        h5.attrs["feature_frame_seconds"] = (
            config.hop_length / config.target_sr
        )
        h5.attrs["output_segment_seconds"] = (
            2.0
            * config.hop_length
            / config.target_sr
        )
        h5.attrs["clip_threshold_train"] = (
            np.nan
            if clip_threshold is None
            else clip_threshold
        )
        h5.attrs["normalization"] = (
            "TRAIN-only per-feature z-score"
        )
        h5.attrs["spectrogram_representation"] = (
            "10*log10(power), then TRAIN-only z-score"
        )
        h5.attrs["config_json"] = json.dumps(
            asdict(config),
            ensure_ascii=False,
            sort_keys=True,
        )

        def flush() -> None:
            nonlocal written
            if not waveform_batch:
                return

            raw_features = extractor.transform_raw(
                np.stack(waveform_batch, axis=0)
            )
            standardized = (
                raw_features
                - feature_mean.reshape(1, 1, -1)
            ) / feature_std.reshape(1, 1, -1)

            standardized = np.clip(
                standardized,
                -12.0,
                12.0,
            ).astype(np.float16)

            count = len(metadata_batch)
            destination = slice(
                written,
                written + count,
            )

            datasets["features"][destination] = standardized
            datasets["frame_target"][destination] = np.stack(
                [
                    item["frame_target"]
                    for item in metadata_batch
                ]
            )
            datasets["target"][destination] = np.stack(
                [
                    item["target"]
                    for item in metadata_batch
                ]
            )
            datasets["label_presence"][destination] = np.stack(
                [
                    item["label_presence"]
                    for item in metadata_batch
                ]
            )

            for key in (
                "source_file",
                "label_file",
                "subject_id",
                "original_sample_rate",
                "recording_duration_sec",
                "peak_before_clip",
                "clipping_ratio",
            ):
                datasets[key][destination] = [
                    item[key]
                    for item in metadata_batch
                ]

            written += count
            waveform_batch.clear()
            metadata_batch.clear()

        for wav_path in tqdm(
            wavs,
            desc=f"Pass 3/3: write {split_name}",
        ):
            label_path = choose_label_path(
                wav_path,
                label_index,
            )

            if label_path is None:
                missing_labels.append(str(wav_path))
                if not config.allow_missing_labels:
                    continue
                events: list[Event] = []
            else:
                try:
                    events = parse_label_file(
                        label_path
                    )
                except Exception as exc:
                    failures.append(
                        {
                            "file": str(wav_path),
                            "label": str(label_path),
                            "error": repr(exc),
                        }
                    )
                    LOGGER.exception(
                        "Label parsing failed: %s",
                        label_path,
                    )
                    continue

            try:
                waveform, original_sr, original_duration = (
                    load_recording(
                        wav_path,
                        config.target_sr,
                        config.record_sec,
                        config.pad_short,
                    )
                )
                waveform = filter_recording(
                    waveform,
                    sos,
                )
                waveform, peak_before_clip, clipping_ratio = (
                    apply_clip(
                        waveform,
                        clip_threshold,
                    )
                )

                frame_target, target = build_labels(
                    events,
                    config,
                    extractor.num_frames,
                )

                waveform_batch.append(waveform)
                metadata_batch.append(
                    {
                        "frame_target": frame_target,
                        "target": target,
                        "label_presence": target.max(
                            axis=0
                        ).astype(np.uint8),
                        "source_file": str(
                            wav_path.relative_to(
                                Path(config.raw_root)
                            )
                        ),
                        "label_file": (
                            ""
                            if label_path is None
                            else str(
                                label_path.relative_to(
                                    Path(config.raw_root)
                                )
                            )
                        ),
                        "subject_id": infer_subject_id(
                            wav_path,
                            split_dir,
                            config.subject_regex,
                        ),
                        "original_sample_rate": original_sr,
                        "recording_duration_sec": original_duration,
                        "peak_before_clip": peak_before_clip,
                        "clipping_ratio": clipping_ratio,
                    }
                )

                if len(waveform_batch) >= config.batch_size:
                    flush()

            except Exception as exc:
                failures.append(
                    {
                        "file": str(wav_path),
                        "error": repr(exc),
                    }
                )
                LOGGER.exception(
                    "Preprocessing failed: %s",
                    wav_path,
                )

        flush()
        resize_datasets(
            datasets,
            written,
        )
        h5.attrs["sample_count"] = written
        h5.flush()

    return {
        "split": split_name,
        "split_dir": str(split_dir),
        "output_path": str(output_path),
        "wav_count": len(wavs),
        "written_samples": written,
        "missing_label_count": len(missing_labels),
        "missing_label_examples": missing_labels[:50],
        "failure_count": len(failures),
        "failures": failures[:100],
    }


def main() -> None:
    args = parse_args()
    configure_logging(args.out_root)

    config = Config(
        raw_root=str(args.raw_root.resolve()),
        out_root=str(args.out_root.resolve()),
        target_sr=args.target_sr,
        record_sec=args.record_sec,
        highpass_hz=args.highpass_hz,
        highpass_order=args.highpass_order,
        n_fft=args.n_fft,
        win_length=args.win_length,
        hop_length=args.hop_length,
        n_mfcc=args.n_mfcc,
        n_mels=args.n_mels,
        delta_width=args.delta_width,
        clip_percentile=args.clip_percentile,
        batch_size=args.batch_size,
        device=args.device,
        compression=args.compression,
        allow_missing_labels=args.allow_missing_labels,
        pad_short=args.pad_short,
        subject_regex=args.subject_regex,
        random_seed=args.seed,
    )

    raw_root = Path(config.raw_root)
    out_root = Path(config.out_root)

    if not raw_root.is_dir():
        raise FileNotFoundError(
            f"Raw root not found: {raw_root}"
        )

    train_dir = resolve_split_dir(
        raw_root,
        "train",
    )
    test_dir = resolve_split_dir(
        raw_root,
        "test",
    )

    matching_report = {
        "train": inspect_matching(train_dir),
        "test": inspect_matching(test_dir),
    }
    matching_path = (
        out_root / "matching_report.json"
    )
    matching_path.write_text(
        json.dumps(
            matching_report,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    LOGGER.info(
        "TRAIN matching: %s",
        matching_report["train"],
    )
    LOGGER.info(
        "TEST matching: %s",
        matching_report["test"],
    )

    if args.dry_run:
        LOGGER.info(
            "Dry run completed. No HDF5 written."
        )
        return

    sos = make_highpass_sos(config)
    extractor = FeatureExtractor(config)

    if args.no_clip:
        clip_threshold = None
        clip_failures: list[dict[str, str]] = []
        LOGGER.info(
            "Robust waveform clipping disabled."
        )
    else:
        clip_threshold, clip_failures = (
            estimate_clip_threshold(
                train_dir,
                config,
                sos,
            )
        )
        LOGGER.info(
            "TRAIN %.5g percentile clipping threshold: %.8g",
            config.clip_percentile,
            clip_threshold,
        )

    feature_mean, feature_std, stats_report = (
        collect_train_feature_stats(
            train_dir=train_dir,
            config=config,
            sos=sos,
            clip_threshold=clip_threshold,
            extractor=extractor,
        )
    )

    LOGGER.info(
        "Feature statistics: count=%d dim=%d std_min=%.6g std_max=%.6g",
        stats_report["frame_count"],
        stats_report["feature_dim"],
        stats_report["std_min"],
        stats_report["std_max"],
    )

    train_output = (
        out_root
        / "HF_Lung_V1_train_15s_logstd.h5"
    )
    test_output = (
        out_root
        / "HF_Lung_V1_test_15s_logstd.h5"
    )

    train_report = write_split(
        split_name="train",
        split_dir=train_dir,
        output_path=train_output,
        config=config,
        sos=sos,
        clip_threshold=clip_threshold,
        extractor=extractor,
        feature_mean=feature_mean,
        feature_std=feature_std,
        overwrite=args.overwrite,
    )
    test_report = write_split(
        split_name="test",
        split_dir=test_dir,
        output_path=test_output,
        config=config,
        sos=sos,
        clip_threshold=clip_threshold,
        extractor=extractor,
        feature_mean=feature_mean,
        feature_std=feature_std,
        overwrite=args.overwrite,
    )

    report = {
        "config": asdict(config),
        "clip_threshold_train": clip_threshold,
        "clip_failure_count": len(clip_failures),
        "clip_failures": clip_failures[:100],
        "feature_stats": stats_report,
        "expected_shapes": {
            "features": [
                "N",
                extractor.num_frames,
                extractor.feature_dim,
            ],
            "frame_target": [
                "N",
                extractor.num_frames,
                2,
            ],
            "target": [
                "N",
                (extractor.num_frames + 1) // 2,
                2,
            ],
        },
        "train": train_report,
        "test": test_report,
    }

    report_path = (
        out_root / "preprocess_report.json"
    )
    report_path.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    LOGGER.info("Completed.")
    LOGGER.info("TRAIN H5: %s", train_output)
    LOGGER.info("TEST H5 : %s", test_output)
    LOGGER.info("Report  : %s", report_path)


if __name__ == "__main__":
    main()

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch


def compute_linear_spectrogram(
    waveform: torch.Tensor,
    sample_rate: int,
    n_fft: int,
    hop_length: int,
    win_length: int,
):
    window = torch.hann_window(win_length, dtype=waveform.dtype, device=waveform.device)
    spec = torch.stft(
        waveform,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=True,
        pad_mode="reflect",
        return_complex=True,
    )
    power = spec.abs().square()
    db = 10.0 * torch.log10(power.clamp_min(1e-10))
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sample_rate)
    times = np.arange(db.shape[-1], dtype=np.float64) * hop_length / sample_rate
    return db.cpu().numpy(), freqs, times


def _resample_binary(values: np.ndarray, n: int) -> np.ndarray:
    values = np.asarray(values).reshape(-1)
    if len(values) == n:
        return values
    if len(values) == 0:
        return np.zeros(n, dtype=np.float32)
    return np.interp(
        np.linspace(0.0, 1.0, n),
        np.linspace(0.0, 1.0, len(values)),
        values,
    )


def save_inference_figure(
    output_path: Path,
    wav_name: str,
    waveform: torch.Tensor,
    sample_rate: int,
    linear_spec_db: np.ndarray,
    spec_freqs: np.ndarray,
    spec_times: np.ndarray,
    feature: np.ndarray,
    feature_times: np.ndarray,
    feature_title: str,
    prob_times: np.ndarray,
    i_prob: np.ndarray,
    e_prob: np.ndarray,
    i_threshold: float,
    e_threshold: float,
    true_i: Optional[np.ndarray] = None,
    true_e: Optional[np.ndarray] = None,
    max_display_hz: Optional[float] = 2000.0,
    dpi: int = 160,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    duration = len(waveform) / sample_rate
    waveform_times = np.arange(len(waveform), dtype=np.float64) / sample_rate
    has_true = true_i is not None and true_e is not None
    rows = ["True I", "True E", "Pred I", "Pred E"] if has_true else ["Pred I", "Pred E"]

    i_pred = (i_prob >= i_threshold).astype(np.float32)
    e_pred = (e_prob >= e_threshold).astype(np.float32)
    heat_rows = []
    if has_true:
        heat_rows.append(_resample_binary(np.asarray(true_i), len(prob_times)))
        heat_rows.append(_resample_binary(np.asarray(true_e), len(prob_times)))
    heat_rows.extend([i_pred, e_pred])
    heat = np.stack(heat_rows, axis=0)

    fig = plt.figure(figsize=(18, 15))
    gs = fig.add_gridspec(
        nrows=5,
        ncols=1,
        height_ratios=[1.0, 3.2, 2.3, 1.3, 2.2],
        hspace=0.18,
    )

    ax0 = fig.add_subplot(gs[0])
    ax0.plot(waveform_times, waveform.cpu().numpy(), linewidth=0.7)
    ax0.set_xlim(0, duration)
    ax0.set_title(f"Inference | {wav_name} | exact 15 s training preprocessing")
    ax0.set_ylabel("Amplitude")
    ax0.tick_params(labelbottom=False)

    ax1 = fig.add_subplot(gs[1], sharex=ax0)
    fmax = min(float(max_display_hz or sample_rate / 2), sample_rate / 2)
    freq_mask = spec_freqs <= fmax
    img1 = ax1.imshow(
        linear_spec_db[freq_mask],
        origin="lower",
        aspect="auto",
        extent=[
            spec_times[0] if len(spec_times) else 0.0,
            spec_times[-1] if len(spec_times) else duration,
            spec_freqs[freq_mask][0],
            spec_freqs[freq_mask][-1],
        ],
        interpolation="nearest",
    )
    ax1.set_ylabel("Frequency (Hz)")
    ax1.tick_params(labelbottom=False)
    cbar = fig.colorbar(img1, ax=ax1, pad=0.01)
    cbar.set_label("Magnitude (dB)")

    ax2 = fig.add_subplot(gs[2], sharex=ax0)
    ax2.imshow(
        feature,
        origin="lower",
        aspect="auto",
        extent=[
            feature_times[0] if len(feature_times) else 0.0,
            feature_times[-1] if len(feature_times) else duration,
            0,
            feature.shape[0] - 1,
        ],
        interpolation="nearest",
    )
    ax2.set_title(feature_title)
    ax2.set_ylabel("Feature bin")
    ax2.tick_params(labelbottom=False)

    ax3 = fig.add_subplot(gs[3], sharex=ax0)
    ax3.imshow(
        heat,
        origin="upper",
        aspect="auto",
        extent=[0, duration, len(rows), 0],
        interpolation="nearest",
        vmin=0.0,
        vmax=1.0,
    )
    ax3.set_yticks(np.arange(len(rows)) + 0.5)
    ax3.set_yticklabels(rows)
    ax3.tick_params(labelbottom=False)

    ax4 = fig.add_subplot(gs[4], sharex=ax0)
    ax4.plot(prob_times, i_prob, label="I probability", linewidth=1.2)
    ax4.plot(prob_times, e_prob, label="E probability", linewidth=1.2)
    ax4.axhline(i_threshold, linestyle="--", linewidth=1.0, label=f"I threshold={i_threshold:.3f}")
    ax4.axhline(e_threshold, linestyle=":", linewidth=1.0, label=f"E threshold={e_threshold:.3f}")
    ax4.set_ylim(-0.02, 1.02)
    ax4.set_xlim(0, duration)
    ax4.set_xlabel("Time (seconds)")
    ax4.set_ylabel("Probability")
    ax4.legend(loc="lower center", ncol=4, frameon=True)

    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

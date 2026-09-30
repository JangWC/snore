from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from scipy import signal


CATEGORY_NAMES = {
    0: "background_only",
    1: "i_only",
    2: "e_only",
    3: "both_ie",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize full-resolution spectral Transformer predictions."
        )
    )
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--split-file", required=True)
    parser.add_argument("--raw-root", required=True)
    parser.add_argument("--row", type=int, default=None)
    parser.add_argument(
        "--select",
        choices=[
            "random",
            "worst",
            "background_worst",
            "i_only_worst",
            "e_only_worst",
            "max_predicted_overlap",
        ],
        default="worst",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save", default=None)
    return parser.parse_args()


def decode_string(value):
    if isinstance(value, bytes):
        return value.decode(
            "utf-8",
            errors="replace",
        )
    return str(value)


def choose_row(
    targets: np.ndarray,
    probabilities: np.ndarray,
    thresholds: np.ndarray,
    categories: np.ndarray,
    mode: str,
    seed: int,
) -> int:
    rng = np.random.default_rng(seed)

    if mode == "random":
        return int(
            rng.integers(0, len(targets))
        )

    predictions = (
        probabilities
        >= thresholds.reshape(1, 1, 2)
    )
    truth = targets > 0.5

    error = np.mean(
        predictions != truth,
        axis=(1, 2),
    )
    overlap = np.mean(
        np.logical_and(
            predictions[..., 0],
            predictions[..., 1],
        ),
        axis=1,
    )

    if mode == "worst":
        return int(np.argmax(error))

    if mode == "max_predicted_overlap":
        return int(np.argmax(overlap))

    category_id = {
        "background_worst": 0,
        "i_only_worst": 1,
        "e_only_worst": 2,
    }[mode]

    candidates = np.flatnonzero(
        categories == category_id
    )
    if len(candidates) == 0:
        raise ValueError(
            f"No samples for {mode}."
        )

    return int(
        candidates[
            np.argmax(error[candidates])
        ]
    )


def main() -> None:
    args = parse_args()

    with np.load(
        args.predictions,
        allow_pickle=False,
    ) as data:
        probabilities = data["probabilities"]
        targets = data["targets"]
        categories = data["categories"]
        sources = data["sources"]
        source_indices = data["source_indices"]
        thresholds = data["thresholds"]
        segment_seconds = float(
            data["segment_seconds"].item()
            if "segment_seconds" in data.files
            else 15.0 / targets.shape[1]
        )

    with np.load(
        args.split_file,
        allow_pickle=False,
    ) as split:
        h5_paths = {
            0: Path(
                str(split["train_h5"].item())
            ),
            1: Path(
                str(split["test_h5"].item())
            ),
        }

    row = (
        int(args.row)
        if args.row is not None
        else choose_row(
            targets,
            probabilities,
            thresholds,
            categories,
            args.select,
            args.seed,
        )
    )

    source = int(sources[row])
    source_index = int(
        source_indices[row]
    )
    h5_path = h5_paths[source]

    with h5py.File(h5_path, "r") as h5:
        source_file = decode_string(
            h5["source_file"][source_index]
        )
        features = np.asarray(
            h5["features"][source_index],
            dtype=np.float32,
        )
        feature_frame_seconds = float(
            h5.attrs.get(
                "feature_frame_seconds",
                15.0 / features.shape[0],
            )
        )

    raw_path = (
        Path(args.raw_root).resolve()
        / source_file
    )
    if not raw_path.exists():
        raise FileNotFoundError(
            f"Raw WAV not found: {raw_path}"
        )

    waveform, sample_rate = sf.read(
        raw_path,
        dtype="float32",
        always_2d=True,
    )
    waveform = waveform.mean(axis=1)
    waveform = waveform[
        : int(round(15.0 * sample_rate))
    ]
    waveform_time = (
        np.arange(len(waveform))
        / sample_rate
    )

    (
        frequencies,
        spec_time,
        spectrogram,
    ) = signal.spectrogram(
        waveform,
        fs=sample_rate,
        window="hann",
        nperseg=256,
        noverlap=192,
        nfft=256,
        scaling="spectrum",
        mode="magnitude",
    )

    time = (
        np.arange(targets.shape[1])
        + 0.5
    ) * segment_seconds

    prediction = (
        probabilities[row]
        >= thresholds.reshape(1, 2)
    )
    truth = targets[row] > 0.5
    predicted_overlap = np.logical_and(
        prediction[:, 0],
        prediction[:, 1],
    )

    strips = np.stack(
        [
            truth[:, 0],
            truth[:, 1],
            prediction[:, 0],
            prediction[:, 1],
        ],
        axis=0,
    )

    standardized_spec = features[:, :129]
    robust_limit = float(
        np.quantile(
            np.abs(standardized_spec),
            0.99,
        )
    )
    robust_limit = max(
        robust_limit,
        1e-6,
    )

    fig, axes = plt.subplots(
        5,
        1,
        figsize=(18, 15),
        sharex=True,
        gridspec_kw={
            "height_ratios": [
                1.2,
                3.6,
                2.5,
                1.4,
                2.2,
            ]
        },
    )

    axes[0].plot(
        waveform_time,
        waveform,
        linewidth=0.6,
    )
    axes[0].set_ylabel("Amplitude")
    axes[0].set_title(
        f"row={row} | "
        f"category={CATEGORY_NAMES[int(categories[row])]} | "
        f"{source_file}"
    )

    image = axes[1].pcolormesh(
        spec_time,
        frequencies,
        20.0 * np.log10(
            spectrogram + 1e-8
        ),
        shading="auto",
    )
    axes[1].set_ylim(
        0,
        min(2000, sample_rate / 2),
    )
    axes[1].set_ylabel("Frequency (Hz)")
    axes[1].set_title(
        "Raw audio spectrogram"
    )
    fig.colorbar(
        image,
        ax=axes[1],
        label="Magnitude (dB)",
    )

    feature_end = (
        np.arange(
            standardized_spec.shape[0] + 1
        )
        * feature_frame_seconds
    )
    axes[2].imshow(
        standardized_spec.T,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        extent=[
            feature_end[0],
            feature_end[-1],
            0,
            standardized_spec.shape[1],
        ],
        vmin=-robust_limit,
        vmax=robust_limit,
    )
    axes[2].set_ylabel("Frequency bin")
    axes[2].set_title(
        "Stored standardized log-power feature"
    )

    axes[3].imshow(
        strips,
        origin="upper",
        aspect="auto",
        interpolation="nearest",
        extent=[
            0,
            targets.shape[1]
            * segment_seconds,
            4,
            0,
        ],
        vmin=0,
        vmax=1,
    )
    axes[3].set_yticks(
        [0.5, 1.5, 2.5, 3.5]
    )
    axes[3].set_yticklabels(
        [
            "True I",
            "True E",
            "Pred I",
            "Pred E",
        ]
    )
    axes[3].set_title(
        "True and predicted intervals"
    )

    axes[4].plot(
        time,
        probabilities[row, :, 0],
        label="I probability",
    )
    axes[4].plot(
        time,
        probabilities[row, :, 1],
        label="E probability",
    )
    axes[4].axhline(
        thresholds[0],
        linestyle="--",
        label=(
            f"I threshold="
            f"{thresholds[0]:.3f}"
        ),
    )
    axes[4].axhline(
        thresholds[1],
        linestyle=":",
        label=(
            f"E threshold="
            f"{thresholds[1]:.3f}"
        ),
    )
    axes[4].fill_between(
        time,
        0,
        1,
        where=predicted_overlap,
        step="mid",
        alpha=0.2,
        label=(
            "Predicted I and E simultaneously"
        ),
    )
    axes[4].set_ylim(-0.03, 1.03)
    axes[4].set_xlim(
        0,
        min(
            15.0,
            len(waveform) / sample_rate,
        ),
    )
    axes[4].set_xlabel("Time (seconds)")
    axes[4].set_ylabel("Probability")
    axes[4].legend(
        loc="upper right",
        ncol=3,
    )
    axes[4].set_title(
        "Full-resolution I/E probabilities"
    )

    plt.tight_layout()

    if args.save:
        save_path = Path(
            args.save
        ).resolve()
        save_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        fig.savefig(
            save_path,
            dpi=160,
            bbox_inches="tight",
        )
        print("saved:", save_path)

    plt.show()

    print("row:", row)
    print("source_file:", source_file)
    print(
        "category:",
        CATEGORY_NAMES[int(categories[row])],
    )
    print(
        "segment seconds:",
        segment_seconds,
    )
    print(
        "segment error:",
        float(
            np.mean(
                prediction != truth
            )
        ),
    )


if __name__ == "__main__":
    main()

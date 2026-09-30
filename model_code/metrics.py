from __future__ import annotations

from typing import Any

import numpy as np


CHANNEL_NAMES = ("I", "E")
CATEGORY_NAMES = {
    0: "background_only",
    1: "i_only",
    2: "e_only",
    3: "both_ie",
}


def _binary_auc(target: np.ndarray, score: np.ndarray) -> float:
    target = np.asarray(target, dtype=np.int64).reshape(-1)
    score = np.asarray(score, dtype=np.float64).reshape(-1)

    positive = int(target.sum())
    negative = int(len(target) - positive)
    if positive == 0 or negative == 0:
        return float("nan")

    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), dtype=np.float64)
    ranks[order] = np.arange(1, len(score) + 1, dtype=np.float64)

    sorted_score = score[order]
    start = 0
    while start < len(score):
        end = start + 1
        while end < len(score) and sorted_score[end] == sorted_score[start]:
            end += 1
        average_rank = ranks[order[start:end]].mean()
        ranks[order[start:end]] = average_rank
        start = end

    positive_rank_sum = ranks[target == 1].sum()
    return float(
        (
            positive_rank_sum
            - positive * (positive + 1) / 2.0
        )
        / (positive * negative)
    )


def binary_metrics(
    target: np.ndarray,
    probability: np.ndarray,
    threshold: float,
) -> dict[str, float | int]:
    target = np.asarray(target, dtype=np.int64).reshape(-1)
    probability = np.asarray(probability, dtype=np.float64).reshape(-1)
    prediction = probability >= threshold

    positive = target == 1
    negative = ~positive

    tp = int(np.logical_and(prediction, positive).sum())
    tn = int(np.logical_and(~prediction, negative).sum())
    fp = int(np.logical_and(prediction, negative).sum())
    fn = int(np.logical_and(~prediction, positive).sum())

    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
    sensitivity = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    precision = tp / max(tp + fp, 1)
    f1 = (
        2.0 * precision * sensitivity
        / max(precision + sensitivity, 1e-12)
    )

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "precision": float(precision),
        "f1": float(f1),
        "auc": _binary_auc(target, probability),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "positive_rate": float(positive.mean()),
        "predicted_positive_rate": float(prediction.mean()),
    }


def choose_threshold(
    target: np.ndarray,
    probability: np.ndarray,
    metric: str = "accuracy",
    minimum: float = 0.05,
    maximum: float = 0.95,
    steps: int = 91,
) -> tuple[float, dict[str, float | int]]:
    if metric not in {"accuracy", "f1"}:
        raise ValueError("metric must be 'accuracy' or 'f1'.")

    best_threshold = 0.5
    best_metrics = binary_metrics(target, probability, best_threshold)
    best_score = float(best_metrics[metric])

    for threshold in np.linspace(minimum, maximum, steps):
        current = binary_metrics(target, probability, float(threshold))
        score = float(current[metric])
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
            best_metrics = current

    return best_threshold, best_metrics


def evaluate_joint(
    targets: np.ndarray,
    probabilities: np.ndarray,
    thresholds: np.ndarray | list[float] | tuple[float, float],
    categories: np.ndarray | None = None,
) -> dict[str, Any]:
    targets = np.asarray(targets)
    probabilities = np.asarray(probabilities)
    thresholds = np.asarray(thresholds, dtype=np.float64).reshape(2)

    if targets.shape != probabilities.shape or targets.shape[-1] != 2:
        raise ValueError(
            f"Expected matching [N,T,2] arrays, got {targets.shape} and "
            f"{probabilities.shape}"
        )

    predictions = probabilities >= thresholds.reshape(1, 1, 2)
    binary_target = targets > 0.5

    channel_result = {
        name: binary_metrics(
            binary_target[..., channel].reshape(-1),
            probabilities[..., channel].reshape(-1),
            float(thresholds[channel]),
        )
        for channel, name in enumerate(CHANNEL_NAMES)
    }

    result: dict[str, Any] = {
        "I": channel_result["I"],
        "E": channel_result["E"],
        "mean_accuracy": float(
            (channel_result["I"]["accuracy"] + channel_result["E"]["accuracy"])
            / 2.0
        ),
        "mean_f1": float(
            (channel_result["I"]["f1"] + channel_result["E"]["f1"])
            / 2.0
        ),
        "exact_match_accuracy": float(
            np.all(predictions == binary_target, axis=-1).mean()
        ),
        "predicted_overlap_rate": float(
            np.logical_and(predictions[..., 0], predictions[..., 1]).mean()
        ),
        "true_overlap_rate": float(
            np.logical_and(binary_target[..., 0], binary_target[..., 1]).mean()
        ),
        "thresholds": {
            "I": float(thresholds[0]),
            "E": float(thresholds[1]),
        },
    }

    if categories is not None:
        categories = np.asarray(categories, dtype=np.int64)
        category_results: dict[str, Any] = {}

        for category_id, category_name in CATEGORY_NAMES.items():
            mask = categories == category_id
            if not mask.any():
                continue

            category_results[category_name] = {
                "recordings": int(mask.sum()),
                "I": binary_metrics(
                    binary_target[mask, :, 0].reshape(-1),
                    probabilities[mask, :, 0].reshape(-1),
                    float(thresholds[0]),
                ),
                "E": binary_metrics(
                    binary_target[mask, :, 1].reshape(-1),
                    probabilities[mask, :, 1].reshape(-1),
                    float(thresholds[1]),
                ),
                "exact_match_accuracy": float(
                    np.all(
                        predictions[mask] == binary_target[mask],
                        axis=-1,
                    ).mean()
                ),
                "predicted_overlap_rate": float(
                    np.logical_and(
                        predictions[mask, :, 0],
                        predictions[mask, :, 1],
                    ).mean()
                ),
            }

        background_mask = categories == 0
        i_only_mask = categories == 1
        e_only_mask = categories == 2

        error_rates = {
            "i_only_e_false_positive_rate": (
                float(predictions[i_only_mask, :, 1].mean())
                if i_only_mask.any()
                else float("nan")
            ),
            "e_only_i_false_positive_rate": (
                float(predictions[e_only_mask, :, 0].mean())
                if e_only_mask.any()
                else float("nan")
            ),
            "background_i_false_positive_rate": (
                float(predictions[background_mask, :, 0].mean())
                if background_mask.any()
                else float("nan")
            ),
            "background_e_false_positive_rate": (
                float(predictions[background_mask, :, 1].mean())
                if background_mask.any()
                else float("nan")
            ),
        }

        result["by_recording_category"] = category_results
        result["composition_error_rates"] = error_rates

    return result

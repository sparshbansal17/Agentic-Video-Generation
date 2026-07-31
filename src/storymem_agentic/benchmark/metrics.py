from __future__ import annotations

from typing import Iterable


def _safe_divide(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator else None


def detection_metrics(expected: Iterable[str], predicted: Iterable[str]) -> dict[str, object]:
    """Score reviewer fault detection using unique fault labels."""
    truth = set(expected)
    output = set(predicted)
    true_positive = len(truth & output)
    false_positive = len(output - truth)
    false_negative = len(truth - output)
    precision = _safe_divide(true_positive, true_positive + false_positive)
    recall = _safe_divide(true_positive, true_positive + false_negative)
    f1 = None
    if precision is not None and recall is not None and precision + recall:
        f1 = 2 * precision * recall / (precision + recall)
    return {
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def localization_metrics(
    expected_scenes: Iterable[int], predicted_scenes: Iterable[int]
) -> dict[str, object]:
    """Measure whether a repair loop identifies only affected scenes."""
    scores = detection_metrics(
        (str(scene) for scene in expected_scenes),
        (str(scene) for scene in predicted_scenes),
    )
    return {
        "scene_precision": scores["precision"],
        "scene_recall": scores["recall"],
        "scene_f1": scores["f1"],
        "collateral_scenes": scores["false_positive"],
        "missed_scenes": scores["false_negative"],
    }

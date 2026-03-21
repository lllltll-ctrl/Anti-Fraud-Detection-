from __future__ import annotations

from typing import Iterable


def precision_recall_summary(y_true: Iterable[int], y_pred: Iterable[int]) -> dict[str, float]:
    true_values = [int(value) for value in y_true]
    pred_values = [int(value) for value in y_pred]

    if len(true_values) != len(pred_values):
        raise ValueError("y_true and y_pred must have the same length")

    true_positive = sum(1 for truth, pred in zip(true_values, pred_values) if truth == 1 and pred == 1)
    false_positive = sum(1 for truth, pred in zip(true_values, pred_values) if truth == 0 and pred == 1)
    false_negative = sum(1 for truth, pred in zip(true_values, pred_values) if truth == 1 and pred == 0)

    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) else 0.0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) else 0.0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
    }


def binary_f1_score(y_true: Iterable[int], y_pred: Iterable[int]) -> float:
    return precision_recall_summary(y_true, y_pred)["f1_score"]

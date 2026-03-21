from __future__ import annotations

from typing import Iterable


def _f1_score(y_true: list[int], y_pred: list[int]) -> float:
    true_positive = sum(1 for truth, pred in zip(y_true, y_pred) if truth == 1 and pred == 1)
    false_positive = sum(1 for truth, pred in zip(y_true, y_pred) if truth == 0 and pred == 1)
    false_negative = sum(1 for truth, pred in zip(y_true, y_pred) if truth == 1 and pred == 0)

    precision_denominator = true_positive + false_positive
    recall_denominator = true_positive + false_negative

    if precision_denominator == 0 or recall_denominator == 0:
        return 0.0

    precision = true_positive / precision_denominator
    recall = true_positive / recall_denominator

    if precision + recall == 0:
        return 0.0

    return 2 * precision * recall / (precision + recall)


def find_best_threshold(y_true: Iterable[int], y_proba: Iterable[float]) -> dict[str, float]:
    y_true_list = [int(value) for value in y_true]
    y_proba_list = [float(value) for value in y_proba]

    if len(y_true_list) != len(y_proba_list):
        raise ValueError("y_true and y_proba must have the same length")

    best_threshold = 0.5
    best_f1_score = 0.0

    ranked = sorted(zip(y_proba_list, y_true_list), key=lambda item: item[0], reverse=True)
    total_positive = sum(y_true_list)
    true_positive = 0
    false_positive = 0
    false_negative = total_positive

    index = 0
    while index < len(ranked):
        threshold = ranked[index][0]
        batch_true_positive = 0
        batch_false_positive = 0

        while index < len(ranked) and ranked[index][0] == threshold:
            if ranked[index][1] == 1:
                batch_true_positive += 1
            else:
                batch_false_positive += 1
            index += 1

        true_positive += batch_true_positive
        false_positive += batch_false_positive
        false_negative -= batch_true_positive

        precision_denominator = true_positive + false_positive
        recall_denominator = true_positive + false_negative
        if precision_denominator == 0 or recall_denominator == 0:
            continue

        precision = true_positive / precision_denominator
        recall = true_positive / recall_denominator
        if precision + recall == 0:
            continue

        score = 2 * precision * recall / (precision + recall)
        if score > best_f1_score:
            best_threshold = threshold
            best_f1_score = score

    return {"threshold": best_threshold, "f1_score": best_f1_score}

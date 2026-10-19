from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .catalogue import GoldClaim
from app.schemas import Claim, Label


LABELS = (Label.SUPPORTED, Label.CONTRADICTED, Label.UNKNOWN)


@dataclass(frozen=True)
class ScoreReport:
    metrics: dict
    missed_gold_ids: list[str]
    unmatched_prediction_ids: list[str]
    wrong_label_ids: list[str]


def score(gold: Iterable[GoldClaim], predictions: Iterable[Claim]) -> ScoreReport:
    gold_list, prediction_list = list(gold), list(predictions)
    gold_by_span = {(item.answer_id, item.start_char, item.end_char): item for item in gold_list}
    matched: dict[tuple[str, int, int], Claim] = {}
    extra: list[Claim] = []
    for prediction in prediction_list:
        key = (prediction.claim_id.split(":", 1)[0], prediction.start_char, prediction.end_char)
        if key in gold_by_span and key not in matched:
            matched[key] = prediction
        else:
            extra.append(prediction)
    g, p, m = len(gold_list), len(prediction_list), len(matched)
    correct = sum(matched[key].label == item.expected_label for key, item in gold_by_span.items() if key in matched)
    columns = [label.value for label in LABELS] + ["missing_or_error"]
    rows = [label.value for label in LABELS] + ["no_gold"]
    confusion = {row: {column: 0 for column in columns} for row in rows}
    for key, item in gold_by_span.items():
        prediction = matched.get(key)
        if prediction is None:
            confusion[item.expected_label.value]["missing_or_error"] += 1
        else:
            confusion[item.expected_label.value][prediction.label.value] += 1
    for prediction in extra:
        confusion["no_gold"][prediction.label.value] += 1
    per_label = _per_label(gold_list, prediction_list, gold_by_span, matched)
    metrics = {
        "gold_claims": g, "accepted_predictions": p, "exact_span_matches": m,
        "correct_joint_matches": correct,
        "extraction": _prf(m, p, g), "joint": _prf(correct, p, g),
        "gold_claim_coverage": _ratio(correct, g),
        "per_label": per_label,
        "macro_f1": sum(item["f1"] for item in per_label.values()) / len(LABELS),
        "confusion_matrix": confusion,
    }
    missed = [item.claim_id for key, item in gold_by_span.items() if key not in matched]
    wrong = [item.claim_id for key, item in gold_by_span.items()
             if key in matched and matched[key].label != item.expected_label]
    return ScoreReport(metrics, missed, [item.claim_id for item in extra], wrong)


def _per_label(gold: list[GoldClaim], predictions: list[Claim], gold_by_span: dict, matched: dict) -> dict:
    result = {}
    for label in LABELS:
        actual = sum(item.expected_label == label for item in gold)
        predicted = sum(item.label == label for item in predictions)
        tp = sum(item.expected_label == label and matched[key].label == label
                 for key, item in gold_by_span.items() if key in matched)
        result[label.value] = _prf(tp, predicted, actual)
    return result


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _prf(tp: int, predicted: int, actual: int) -> dict[str, float]:
    precision, recall = _ratio(tp, predicted), _ratio(tp, actual)
    return {"precision": precision, "recall": recall,
            "f1": _ratio(2 * precision * recall, precision + recall)}

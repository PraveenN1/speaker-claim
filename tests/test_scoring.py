import pytest

from app.services.catalogue import GoldClaim
from app.schemas import Claim, Label
from app.services.scoring import score


def gold(index: int, label=Label.SUPPORTED):
    return GoldClaim("A01", f"A01:{index}:{index + 1}", index, index + 1, "x", label, ())


def prediction(index: int, label=Label.SUPPORTED):
    return Claim(claim_id=f"A01:{index}:{index + 1}", start_char=index, end_char=index + 1,
                 quote="x", label=label, fact_ids=[], reason="test")


def test_reference_arithmetic_example():
    references = [gold(index, Label.SUPPORTED) for index in range(10)]
    outputs = [prediction(index, Label.SUPPORTED if index < 7 else Label.UNKNOWN) for index in range(8)]
    outputs.extend(prediction(100 + index) for index in range(2))
    report = score(references, outputs)
    assert report.metrics["extraction"]["f1"] == pytest.approx(0.8)
    assert report.metrics["joint"]["f1"] == pytest.approx(0.7)


def test_missed_and_extra_claims_are_accounted_for():
    report = score([gold(0)], [prediction(8)])
    assert report.missed_gold_ids == ["A01:0:1"]
    assert report.unmatched_prediction_ids == ["A01:8:9"]
    assert report.metrics["confusion_matrix"]["supported"]["missing_or_error"] == 1
    assert report.metrics["confusion_matrix"]["no_gold"]["supported"] == 1

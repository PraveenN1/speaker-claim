from pathlib import Path

import pytest

from app.services.catalogue import Dataset
from app.schemas import Claim, ClassifiedAnswer, Label
from app.services.validation import OutputValidationError, validate_output


@pytest.fixture(scope="module")
def dataset():
    return Dataset(Path("."))


def claim(source: str, start: int, *, label=Label.UNKNOWN):
    end = start + len("This is factual.")
    return Claim(claim_id=f"A01:{start}:{end}", start_char=start, end_char=end,
                 quote=source[start:end], label=label, fact_ids=[], reason="No fact establishes it.")


def test_validates_exact_quote_and_span(dataset):
    source = "Advice. This is factual."
    output = ClassifiedAnswer(answer_id="A01", claims=[claim(source, 8)])
    assert validate_output(output, source, dataset) == output


def test_rejects_overlap_and_invented_fact_id(dataset):
    source = "This is factual. This is factual."
    first = claim(source, 0)
    second = claim(source, 5)
    with pytest.raises(OutputValidationError, match="overlap"):
        validate_output(ClassifiedAnswer(answer_id="A01", claims=[first, second]), source, dataset)
    bad = claim(source, 0).model_copy(update={"fact_ids": ["not-real"]})
    with pytest.raises(OutputValidationError, match="invented"):
        validate_output(ClassifiedAnswer(answer_id="A01", claims=[bad]), source, dataset)

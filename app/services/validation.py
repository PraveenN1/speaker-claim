from __future__ import annotations

from .catalogue import Dataset
from app.schemas import Claim, ClassifiedAnswer


class OutputValidationError(ValueError):
    """A provider response failed the public output contract."""


def validate_output(output: ClassifiedAnswer, source: str, dataset: Dataset) -> ClassifiedAnswer:
    seen_ids: set[str] = set()
    previous_end = 0
    known_fact_ids = {fact.fact_id for fact in dataset.facts}
    for claim in output.claims:
        _validate_claim(claim, output.answer_id, source, known_fact_ids)
        if claim.claim_id in seen_ids:
            raise OutputValidationError(f"duplicate claim_id: {claim.claim_id}")
        seen_ids.add(claim.claim_id)
        if claim.start_char < previous_end:
            raise OutputValidationError("claims overlap or are not in increasing span order")
        previous_end = claim.end_char
    return output


def _validate_claim(claim: Claim, answer_id: str, source: str, known_fact_ids: set[str]) -> None:
    if claim.end_char <= claim.start_char or claim.end_char > len(source):
        raise OutputValidationError(f"invalid span for {claim.claim_id}")
    expected_id = f"{answer_id}:{claim.start_char}:{claim.end_char}"
    if claim.claim_id != expected_id:
        raise OutputValidationError(f"claim_id must be {expected_id}")
    if claim.quote != source[claim.start_char:claim.end_char]:
        raise OutputValidationError(f"quote does not match span for {claim.claim_id}")
    if claim.quote != claim.quote.strip():
        raise OutputValidationError(f"quote has surrounding whitespace for {claim.claim_id}")
    unknown_fact_ids = set(claim.fact_ids) - known_fact_ids
    if unknown_fact_ids:
        raise OutputValidationError(f"invented fact_ids: {sorted(unknown_fact_ids)}")

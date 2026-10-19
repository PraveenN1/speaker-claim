from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from app.schemas import AgentInput, Label, Split


@dataclass(frozen=True)
class Fact:
    fact_id: str
    brand_id: str
    model_id: str
    field: str
    value: str
    value_type: str
    unit: str
    condition: str
    valid_from: date | None
    valid_to: date | None


@dataclass(frozen=True)
class GoldClaim:
    answer_id: str
    claim_id: str
    start_char: int
    end_char: int
    quote: str
    expected_label: Label
    evidence_fact_ids: tuple[str, ...]


def _read_csv(path: Path) -> list[dict[str, str]]:
    # newline="" is essential: embedded newlines are valid CSV field contents.
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


class Dataset:
    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir
        answer_rows = _read_csv(data_dir / "answers.csv")
        fact_rows = _read_csv(data_dir / "brand_facts.csv")
        gold_rows = _read_csv(data_dir / "development_gold.csv")
        self.answers = {
            row["answer_id"]: AgentInput(
                answer_id=row["answer_id"], brand_id=row["brand_id"],
                question=row["question"], ai_answer=row["ai_answer"]
            )
            for row in answer_rows
        }
        self.splits = {row["answer_id"]: Split(row["split"]) for row in answer_rows}
        self.facts = [
            Fact(
                fact_id=row["fact_id"], brand_id=row["brand_id"], model_id=row["model_id"],
                field=row["field"], value=row["value"], value_type=row["value_type"],
                unit=row["unit"], condition=row["condition"],
                valid_from=_date(row["valid_from"]), valid_to=_date(row["valid_to"]),
            )
            for row in fact_rows
        ]
        self.facts_by_brand: dict[str, list[Fact]] = {}
        for fact in self.facts:
            self.facts_by_brand.setdefault(fact.brand_id, []).append(fact)
        self.gold = [
            GoldClaim(
                answer_id=row["answer_id"], claim_id=row["claim_id"],
                start_char=int(row["start_char"]), end_char=int(row["end_char"]), quote=row["quote"],
                expected_label=Label(row["expected_label"]),
                evidence_fact_ids=tuple(filter(None, row["evidence_fact_ids"].split(","))),
            )
            for row in gold_rows
        ]
        self.gold_by_answer: dict[str, list[GoldClaim]] = {}
        for claim in self.gold:
            self.gold_by_answer.setdefault(claim.answer_id, []).append(claim)
        self.file_hashes = {name: _sha256(data_dir / name) for name in (
            "answers.csv", "brand_facts.csv", "development_gold.csv"
        )}

    def inputs_for_split(self, split: Split) -> list[AgentInput]:
        return sorted(
            (self.answers[key] for key, value in self.splits.items() if value == split),
            key=lambda answer: answer.answer_id,
        )

    def get_brand_facts(self, brand_id: str) -> list[Fact]:
        if brand_id not in self.facts_by_brand:
            raise KeyError(brand_id)
        return self.facts_by_brand[brand_id]


def _date(value: str) -> date | None:
    return date.fromisoformat(value) if value else None


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

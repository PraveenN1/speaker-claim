from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Label(StrEnum):
    SUPPORTED = "supported"
    CONTRADICTED = "contradicted"
    UNKNOWN = "unknown"


class AgentVersion(StrEnum):
    V1 = "v1"
    V2 = "v2"


class Split(StrEnum):
    DEVELOPMENT = "development"
    CHECK = "check"


class AgentInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer_id: str = Field(min_length=1)
    brand_id: str = Field(min_length=1)
    question: str
    ai_answer: str


class Claim(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claim_id: str
    start_char: int = Field(ge=0)
    end_char: int = Field(ge=0)
    quote: str = Field(min_length=1)
    label: Label
    fact_ids: list[str]
    reason: str = Field(max_length=1_500)


class ClassifiedAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer_id: str
    claims: list[Claim]


class ClassifyRequest(AgentInput):
    agent_version: AgentVersion

    @field_validator("answer_id", "brand_id")
    @classmethod
    def no_surrounding_space(cls, value: str) -> str:
        if value != value.strip():
            raise ValueError("must not have leading or trailing whitespace")
        return value


class EvaluationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    agent_version: AgentVersion
    split: Split


class ErrorPayload(BaseModel):
    code: str
    message: str


class AnswerOutcome(BaseModel):
    answer_id: str
    status: str
    latency_ms: int | None = None
    error: ErrorPayload | None = None
    claims: list[Claim] = Field(default_factory=list)


class RunReport(BaseModel):
    run_id: str
    split: Split
    agent_version: AgentVersion
    status: str
    counts: dict[str, int]
    metrics: dict[str, Any] | None

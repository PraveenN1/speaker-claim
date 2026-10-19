from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import JSON, ForeignKey, Integer, String, Text, UniqueConstraint, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, relationship

from app.schemas import AnswerOutcome, Claim, Split


class Base(DeclarativeBase):
    pass


class Run(Base):
    __tablename__ = "evaluation_runs"
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    split: Mapped[str] = mapped_column(String(16), nullable=False)
    agent_version: Mapped[str] = mapped_column(String(8), nullable=False)
    status: Mapped[str] = mapped_column(String(24), nullable=False)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    metrics_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[str] = mapped_column(String(40), nullable=False)
    answers: Mapped[list[AnswerResult]] = relationship(back_populates="run", cascade="all, delete-orphan")


class AnswerResult(Base):
    __tablename__ = "answer_results"
    __table_args__ = (UniqueConstraint("run_id", "answer_id", name="uq_run_answer"),)
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("evaluation_runs.id"), nullable=False)
    answer_id: Mapped[str] = mapped_column(String(32), nullable=False)
    status: Mapped[str] = mapped_column(String(24), nullable=False)
    latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    error_json: Mapped[dict[str, str] | None] = mapped_column(JSON, nullable=True)
    run: Mapped[Run] = relationship(back_populates="answers")
    claims: Mapped[list[ClaimResult]] = relationship(back_populates="answer", cascade="all, delete-orphan")


class ClaimResult(Base):
    __tablename__ = "claim_results"
    __table_args__ = (UniqueConstraint("run_id", "answer_id", "start_char", "end_char", name="uq_run_span"),)
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("evaluation_runs.id"), nullable=False)
    answer_id: Mapped[str] = mapped_column(String(32), nullable=False)
    answer_result_id: Mapped[int] = mapped_column(ForeignKey("answer_results.id"), nullable=False)
    claim_id: Mapped[str] = mapped_column(String(80), nullable=False)
    start_char: Mapped[int] = mapped_column(Integer, nullable=False)
    end_char: Mapped[int] = mapped_column(Integer, nullable=False)
    quote: Mapped[str] = mapped_column(Text, nullable=False)
    label: Mapped[str] = mapped_column(String(16), nullable=False)
    fact_ids: Mapped[list[str]] = mapped_column(JSON, nullable=False)
    reason: Mapped[str] = mapped_column(Text, nullable=False)
    answer: Mapped[AnswerResult] = relationship(back_populates="claims")


def make_engine(url: str):
    return create_engine(url, future=True)


def initialize(engine) -> None:
    Base.metadata.create_all(engine)


class RunStore:
    def __init__(self, engine) -> None:
        self.engine = engine

    def create_run(self, split: Split, agent_version: str, metadata: dict[str, Any]) -> str:
        run_id = str(uuid.uuid4())
        with Session(self.engine) as session:
            session.add(Run(id=run_id, split=split.value, agent_version=agent_version, status="processing",
                            metadata_json=metadata, metrics_json=None,
                            created_at=datetime.now(timezone.utc).isoformat()))
            session.commit()
        return run_id

    def save_outcome(self, run_id: str, outcome: AnswerOutcome) -> None:
        with Session(self.engine) as session:
            result = AnswerResult(run_id=run_id, answer_id=outcome.answer_id, status=outcome.status,
                                  latency_ms=outcome.latency_ms,
                                  error_json=outcome.error.model_dump() if outcome.error else None)
            session.add(result)
            session.flush()
            for claim in outcome.claims:
                session.add(ClaimResult(run_id=run_id, answer_id=outcome.answer_id, answer_result_id=result.id,
                    claim_id=claim.claim_id, start_char=claim.start_char, end_char=claim.end_char,
                    quote=claim.quote, label=claim.label.value, fact_ids=claim.fact_ids, reason=claim.reason))
            session.commit()

    def finish(self, run_id: str, status: str, metrics: dict[str, Any] | None) -> None:
        with Session(self.engine) as session:
            run = session.get(Run, run_id)
            if run is None:
                raise KeyError(run_id)
            run.status, run.metrics_json = status, metrics
            session.commit()

    def get_run(self, run_id: str) -> Run | None:
        with Session(self.engine) as session:
            run = session.get(Run, run_id)
            if run is not None:
                # Materialize before the session closes.
                _ = [(answer.answer_id, list(answer.claims)) for answer in run.answers]
            return run

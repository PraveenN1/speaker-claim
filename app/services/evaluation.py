from __future__ import annotations

import asyncio
import time

from .classification import AgentFailure, ClaimAgent
from .catalogue import Dataset
from app.repositories.database import RunStore
from app.schemas import AgentVersion, AnswerOutcome, ErrorPayload, Split
from .scoring import score


class EvaluationHarness:
    def __init__(self, dataset: Dataset, agent: ClaimAgent, store: RunStore) -> None:
        self.dataset, self.agent, self.store = dataset, agent, store

    async def run(self, split: Split, version: AgentVersion) -> str:
        run_id = self.store.create_run(split, version.value, {"file_hashes": self.dataset.file_hashes})
        semaphore = asyncio.Semaphore(3)
        async def one(item):
            started = time.perf_counter()
            try:
                async with semaphore:
                    output = await self.agent.classify(item, version)
                result = AnswerOutcome(answer_id=item.answer_id, status="accepted",
                    latency_ms=round((time.perf_counter() - started) * 1000), claims=output.claims)
            except AgentFailure as error:
                result = AnswerOutcome(answer_id=item.answer_id, status="error",
                    latency_ms=round((time.perf_counter() - started) * 1000),
                    error=ErrorPayload(code="agent_error", message=str(error)))
            self.store.save_outcome(run_id, result)
            return result
        outcomes = await asyncio.gather(*(one(item) for item in self.dataset.inputs_for_split(split)))
        metrics = None
        if split == Split.DEVELOPMENT:
            accepted = [claim for outcome in outcomes if outcome.status == "accepted" for claim in outcome.claims]
            report = score(self.dataset.gold, accepted)
            metrics = {**report.metrics, "missed_gold_ids": report.missed_gold_ids,
                       "unmatched_prediction_ids": report.unmatched_prediction_ids,
                       "matched_wrong_label_ids": report.wrong_label_ids,
                       "answer_error_count": sum(item.status == "error" for item in outcomes)}
        self.store.finish(run_id, "completed", metrics)
        return run_id

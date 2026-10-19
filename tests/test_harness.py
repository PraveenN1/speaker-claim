import asyncio
from pathlib import Path

import pytest

from app.services.classification import ClaimAgent
from app.services.catalogue import Dataset
from app.repositories.database import RunStore, initialize, make_engine
from app.services.evaluation import EvaluationHarness
from app.schemas import AgentVersion, Split


class EmptyAsyncProvider:
    def __init__(self):
        self.active = 0
        self.maximum_active = 0

    async def complete(self, messages, tools):
        self.active += 1
        self.maximum_active = max(self.maximum_active, self.active)
        await asyncio.sleep(0.005)
        self.active -= 1
        return {"content": '{"claims": []}'}


@pytest.mark.asyncio
async def test_harness_bounds_concurrency_and_persists_all_scheduled_answers(tmp_path):
    dataset = Dataset(Path("."))
    provider = EmptyAsyncProvider()
    engine = make_engine(f"sqlite:///{tmp_path / 'runs.db'}")
    initialize(engine)
    store = RunStore(engine)
    run_id = await EvaluationHarness(dataset, ClaimAgent(provider, dataset), store).run(
        Split.DEVELOPMENT, AgentVersion.V1)
    run = store.get_run(run_id)
    assert provider.maximum_active == 3
    assert len(run.answers) == 40
    assert run.status == "completed"
    assert run.metrics_json["gold_claims"] == 376
    assert run.metrics_json["answer_error_count"] == 0

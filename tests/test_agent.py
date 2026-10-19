import json
from pathlib import Path

import pytest

from app.services.classification import AgentFailure, ClaimAgent
from app.services.catalogue import Dataset
from app.schemas import AgentInput, AgentVersion


class FakeProvider:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.messages = []

    async def complete(self, messages, tools):
        self.messages.append(messages)
        return next(self.replies)


@pytest.fixture(scope="module")
def dataset():
    return Dataset(Path("."))


@pytest.mark.asyncio
async def test_uses_catalogue_tool_then_materializes_quote_offsets(dataset):
    text = "The Summit One has a 12-hour battery life. Choose what suits you."
    quote = "The Summit One has a 12-hour battery life."
    provider = FakeProvider([
        {"tool_calls": [{"id": "tool-1", "function": {
            "name": "get_brand_facts", "arguments": '{"brand_id":"cleartrail"}'}}]},
        {"content": json.dumps({"claims": [{"quote": quote, "label": "unknown", "fact_ids": [], "reason": "No matching fact."}]})},
    ])
    output = await ClaimAgent(provider, dataset).classify(
        AgentInput(answer_id="T01", brand_id="cleartrail", question="q", ai_answer=text), AgentVersion.V1)
    assert output.claims[0].start_char == 0
    assert output.claims[0].end_char == len(quote)
    assert output.claims[0].claim_id == f"T01:0:{len(quote)}"
    tool_message = provider.messages[1][-1]
    assert tool_message["role"] == "tool"
    assert "fact_id" in tool_message["content"]


@pytest.mark.asyncio
async def test_illegal_tool_and_invented_quote_fail_the_answer(dataset):
    item = AgentInput(answer_id="T01", brand_id="cleartrail", question="q", ai_answer="Only advice.")
    bad_tool = FakeProvider([{"tool_calls": [{"id": "x", "function": {"name": "drop_database", "arguments": "{}"}}]}])
    with pytest.raises(AgentFailure, match="illegal_tool"):
        await ClaimAgent(bad_tool, dataset).classify(item, AgentVersion.V1)
    bad_quote = FakeProvider([{"content": '{"claims":[{"quote":"invented.","label":"unknown","fact_ids":[],"reason":"x"}]}' }])
    with pytest.raises(AgentFailure, match="malformed_provider_output"):
        await ClaimAgent(bad_quote, dataset).classify(item, AgentVersion.V1)

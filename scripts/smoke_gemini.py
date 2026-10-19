"""Run one live, non-persistent Gemini classification smoke test."""

import asyncio
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.classification import ClaimAgent
from app.services.catalogue import Dataset
from app.providers.gemini import GeminiProvider
from app.schemas import AgentVersion


async def main() -> None:
    dataset = Dataset(Path("."))
    item = dataset.answers["A01"]
    print({"status": "starting", "answer_id": item.answer_id}, flush=True)
    output = await ClaimAgent(GeminiProvider(), dataset).classify(item, AgentVersion.V1)
    print({"answer_id": output.answer_id, "claims": len(output.claims),
           "labels": [claim.label.value for claim in output.claims]})


if __name__ == "__main__":
    asyncio.run(main())

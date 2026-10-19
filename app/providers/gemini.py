from __future__ import annotations

import json
import os
from typing import Any

from openai import AsyncOpenAI
from dotenv import load_dotenv


class GeminiProvider:
    """Gemini adapter using Google's OpenAI-compatible function-calling endpoint."""

    def __init__(self, model: str | None = None) -> None:
        load_dotenv()
        self.client: AsyncOpenAI | None = None
        self.model = model or os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")

    async def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> dict[str, Any]:
        if self.client is None:
            api_key = os.environ.get("GEMINI_API_KEY")
            if not api_key:
                raise RuntimeError("GEMINI_API_KEY is not configured")
            self.client = AsyncOpenAI(
                api_key=api_key,
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            )
        response = await self.client.chat.completions.create(
            model=self.model, messages=messages, tools=tools, tool_choice="auto", temperature=0,
            reasoning_effort="low",
            response_format={"type": "json_object"},
        )
        message = response.choices[0].message
        # Preserve Gemini-specific fields (including thought signatures) exactly.
        # Reconstructing tool calls from the typed OpenAI objects drops these fields,
        # making Gemini thinking-model tool continuations invalid.
        assistant_message = message.model_dump(exclude_none=True)
        return {
            "content": message.content,
            "tool_calls": assistant_message.get("tool_calls", []),
            "assistant_message": assistant_message,
        }

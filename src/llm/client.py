"""Small LLM client wrapper."""

from __future__ import annotations

import os
from typing import Any


class LLMClient:
    def __init__(
        self,
        provider: str = "mock",
        model: str = "mock-relationship-v1",
        api_key: str | None = None,
    ):
        self.provider = provider
        self.model = model
        self.api_key = api_key

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        if self.provider == "mock":
            return self._generate_mock(user_prompt)
        if self.provider == "openai":
            return self._generate_openai(system_prompt, user_prompt)
        if self.provider == "anthropic":
            return self._generate_anthropic(system_prompt, user_prompt)
        raise ValueError(f"unknown llm provider: {self.provider}")

    def _generate_mock(self, user_prompt: str) -> str:
        question = "the latest question"
        for line in user_prompt.splitlines():
            if line.startswith("Question:"):
                question = line.split("Question:", 1)[1].strip()
                break

        return (
            "Mock answer for now. Based on the retrieved notes, "
            f"the main thing to focus on is: {question}. "
            "This mode is mainly for local backend wiring before real API keys."
        )

    def _generate_openai(self, system_prompt: str, user_prompt: str) -> str:
        from openai import OpenAI

        client = OpenAI(api_key=self.api_key or os.getenv("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        text = response.choices[0].message.content
        if not text:
            raise RuntimeError("openai returned empty content")
        return text

    def _generate_anthropic(self, system_prompt: str, user_prompt: str) -> str:
        from anthropic import Anthropic

        client = Anthropic(api_key=self.api_key or os.getenv("ANTHROPIC_API_KEY"))
        response = client.messages.create(
            model=self.model,
            max_tokens=500,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        parts = [item.text for item in response.content if getattr(item, "type", "") == "text"]
        text = "\n".join(parts).strip()
        if not text:
            raise RuntimeError("anthropic returned empty content")
        return text


def build_llm_client(config: dict[str, Any]) -> LLMClient:
    llm_config = dict(config.get("llm", {}))
    return LLMClient(
        provider=str(llm_config.get("provider", "mock")),
        model=str(llm_config.get("model", "mock-relationship-v1")),
        api_key=llm_config.get("api_key"),
    )

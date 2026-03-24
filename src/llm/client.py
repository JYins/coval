"""Small LLM client wrapper."""

from __future__ import annotations

import json
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
        lower_text = user_prompt.lower()
        if "big_five_scores" in lower_text and "topics_of_interest" in lower_text:
            return self._mock_personality_json(user_prompt)
        if "communication_style" in lower_text and "return json only" in lower_text:
            return self._mock_communication_json(user_prompt)
        if "pre-meeting briefing" in lower_text or "before i meet" in lower_text:
            return self._mock_briefing(user_prompt)

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

    def _mock_personality_json(self, user_prompt: str) -> str:
        lower_text = user_prompt.lower()
        topics = []
        preferences = []

        keyword_map = {
            "jazz": "jazz",
            "music": "music",
            "hiking": "hiking",
            "coffee": "coffee",
            "travel": "travel",
            "family": "family",
            "work": "work",
        }
        for word, label in keyword_map.items():
            if word in lower_text:
                topics.append(label)

        if "not noisy" in lower_text or "hate crowded" in lower_text:
            preferences.append("prefers quieter places")
        if "coffee" in lower_text:
            preferences.append("likes coffee chats")
        if "jazz" in lower_text:
            preferences.append("likes music-related hangouts")

        scores = {
            "openness": 0.72 if {"jazz", "music", "travel"} & set(topics) else 0.58,
            "conscientiousness": 0.61,
            "extraversion": 0.67 if "friends" in lower_text or "party" in lower_text else 0.52,
            "agreeableness": 0.68 if "thanks" in lower_text or "help" in lower_text else 0.57,
            "neuroticism": 0.34 if "calm" in lower_text else 0.45,
        }
        return json.dumps(
            {
                "big_five_scores": scores,
                "preferences": preferences,
                "topics_of_interest": topics,
            }
        )

    def _mock_communication_json(self, user_prompt: str) -> str:
        lower_text = user_prompt.lower()
        style = {
            "directness": "direct" if "be direct" in lower_text or "straight" in lower_text else "balanced",
            "tone": "casual" if "haha" in lower_text or "lol" in lower_text else "warm",
            "pace": "fast" if "quick" in lower_text else "steady",
        }
        return json.dumps({"communication_style": style})

    def _mock_briefing(self, user_prompt: str) -> str:
        lower_text = user_prompt.lower()
        hook = "Keep the chat warm and easy."
        if "jazz" in lower_text or "music" in lower_text:
            hook = "Open with music or jazz first, that should land naturally."
        if "hiking" in lower_text:
            hook = "Outdoor plans are a safe opener here."
        return (
            f"{hook} "
            "Use one detail from recent conversations, keep the tone natural, "
            "and avoid jumping too fast into heavy topics."
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

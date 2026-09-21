"""Tests for the small LLM client wrapper."""

from src.llm.client import LLMClient


def test_mock_answer_cites_retrieved_context():
    client = LLMClient(provider="mock")
    prompt = """Retrieved conversation chunks:
[1] score=0.900 chunk=0 text=小明最喜欢桂花乌龙茶。

Question: 小明喜欢喝什么？"""

    answer = client.generate(system_prompt="test", user_prompt=prompt)

    assert "演示模式回答" in answer
    assert "[1] 小明最喜欢桂花乌龙茶。" in answer
    assert "页面下方的引用片段" in answer

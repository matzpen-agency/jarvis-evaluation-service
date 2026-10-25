"""
test_llm_order_judge.py — The LLM order judge against OpenAI-compatible
endpoints: OpenAI, and private-network models (Qwen, gpt-oss).
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

import httpx
import pytest
import respx

from src.infrastructure.llm.llm_order_judge import LlmOrderJudge, parse_decision


def _reply(content=None, reasoning=None) -> httpx.Response:
    message = {"role": "assistant", "content": content}
    if reasoning is not None:
        message["reasoning_content"] = reasoning
    return httpx.Response(200, json={"choices": [{"message": message}]})


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr("asyncio.sleep", AsyncMock())


@pytest.mark.unit
@pytest.mark.parametrize(
    ("content", "expected"),
    [
        ('{"requires_order": true, "reason": "sorted"}', (True, "sorted")),
        ('<think>hmm, "by" means grouping</think>\n{"requires_order": false, "reason": "per group"}', (False, "per group")),
        ('Sure!\n```json\n{"requires_order": "yes", "reason": "rank"}\n```', (True, "rank")),
        ("no json here", None),
        (None, None),
    ],
)
def test_parse_decision(content, expected):
    assert parse_decision(content) == expected


@pytest.mark.unit
@pytest.mark.asyncio
@respx.mock
async def test_openai_request_shape_and_decision():
    route = respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=_reply('{"requires_order": false, "reason": "breakdown per territory"}')
    )
    judge = LlmOrderJudge("https://api.openai.com/v1", "sk-test", "gpt-4.1-mini")

    decision = await judge.requires_order("What is our revenue by sales territory?")

    assert (decision.requires_order, decision.reason, decision.source) == (
        False, "breakdown per territory", "llm",
    )
    request = route.calls.last.request
    assert request.headers["authorization"] == "Bearer sk-test"
    body = json.loads(request.content)
    assert body["model"] == "gpt-4.1-mini"
    assert body["temperature"] == 0
    assert "revenue by sales territory" in body["messages"][1]["content"]


@pytest.mark.unit
@pytest.mark.asyncio
@respx.mock
async def test_private_qwen_model_skips_thinking_and_strips_it():
    route = respx.post("http://ai-service.internal/v1/chat/completions").mock(
        return_value=_reply('<think>...</think>{"requires_order": true, "reason": "מהגבוה לנמוך"}')
    )
    judge = LlmOrderJudge("http://ai-service.internal/v1", "k", "Qwen/Qwen3.6-27B-FP8")

    decision = await judge.requires_order("הצג את המוצרים לפי מחיר מהגבוה לנמוך")

    assert decision.requires_order is True
    assert json.loads(route.calls.last.request.content)["messages"][1]["content"].endswith("/no_think")


@pytest.mark.unit
@pytest.mark.asyncio
@respx.mock
async def test_gpt_oss_answer_in_reasoning_content():
    respx.post("http://ai-service.internal/v1/chat/completions").mock(
        return_value=_reply(content="", reasoning='{"requires_order": false, "reason": "count only"}')
    )
    judge = LlmOrderJudge("http://ai-service.internal/v1", "k", "openai/gpt-oss-120b")
    assert (await judge.requires_order("כמה לקוחות יש?")).requires_order is False


@pytest.mark.unit
@pytest.mark.asyncio
@respx.mock
async def test_retries_without_temperature_when_rejected():
    route = respx.post("https://api.openai.com/v1/chat/completions").mock(
        side_effect=[
            httpx.Response(400, json={"error": {"message": "temperature not supported"}}),
            _reply('{"requires_order": true, "reason": "ranked"}'),
        ]
    )
    judge = LlmOrderJudge("https://api.openai.com/v1", "k", "o4-mini")
    assert (await judge.requires_order("Rank the regions")).requires_order is True
    assert "temperature" not in json.loads(route.calls.last.request.content)


@pytest.mark.unit
@pytest.mark.asyncio
@respx.mock
async def test_unreachable_llm_is_unavailable_and_not_cached(tmp_path):
    route = respx.post("http://ai-service.internal/v1/chat/completions").mock(
        side_effect=httpx.ConnectError("down")
    )
    cache = tmp_path / "judgments.json"
    judge = LlmOrderJudge("http://ai-service.internal/v1", "k", "m", cache_path=str(cache))

    decision = await judge.requires_order("Show sales sorted by date")
    assert decision.requires_order is None
    assert decision.source == "unavailable"
    assert not cache.exists()
    assert route.call_count == 3


@pytest.mark.unit
@pytest.mark.asyncio
@respx.mock
async def test_decisions_are_saved_and_reused_across_runs(tmp_path):
    route = respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=_reply('{"requires_order": true, "reason": "chronological"}')
    )
    cache = tmp_path / "judgments.json"
    first = LlmOrderJudge("https://api.openai.com/v1", "k", "m", cache_path=str(cache))
    await first.requires_order("Monthly sales in chronological order")
    await first.requires_order("Monthly sales in chronological order")

    second = LlmOrderJudge("https://api.openai.com/v1", "k", "m", cache_path=str(cache))
    decision = await second.requires_order("Monthly sales in chronological order")

    assert route.call_count == 1  # judged once, then read from the saved file
    assert (decision.requires_order, decision.source) == (True, "cache")
    saved = json.loads(cache.read_text(encoding="utf-8"))
    assert next(iter(saved.values()))["reason"] == "chronological"

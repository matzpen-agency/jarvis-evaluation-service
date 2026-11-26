"""
test_rate_limit.py — Waiting out the LLM provider's rate limit (HTTP 429):
the policy, the per-question stop switch, the agent and judge retries, and
the stop endpoint.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
import respx

from src.application.jobs.evaluation_job_store import EvaluationJobStore
from src.domain.entities.agent_response import AgentResponse
from src.domain.services.evaluation_engine import EvaluationEngine
from src.domain.services.evaluation_runner import EvaluationRunner
from src.domain.services.metrics_aggregator import MetricsAggregator
from src.domain.services.rate_limit import (
    QuestionScope,
    RateLimitExhaustedError,
    RateLimitPolicy,
    RateLimitStoppedError,
    RateLimitTracker,
    enter_question,
    is_rate_limit,
    leave_question,
    wait_before_retry,
)
from src.infrastructure.llm.llm_order_judge import LlmOrderJudge

FAST = RateLimitPolicy(max_retries=2, wait_seconds=0.01)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (429, True),
        (500, False),
        ("Error code: 429 - {'error': 'Too Many Requests'}", True),
        ("agent tool error: RateLimitError: rate limit reached for gpt-4o", True),
        ("Rate limit exceeded", True),
        ("Resource exhausted: quota exceeded", True),
        ("agent_call_timed_out", False),
        ("returned 1429 rows", False),
        (None, False),
    ],
)
def test_is_rate_limit(error, expected):
    assert is_rate_limit(error) is expected


@pytest.mark.unit
def test_wait_doubles_each_retry():
    policy = RateLimitPolicy(max_retries=3, wait_seconds=20)
    assert [policy.delay(r) for r in (1, 2, 3)] == [20, 40, 80]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_question_waits_is_listed_then_gives_up():
    tracker = RateLimitTracker()
    scope = QuestionScope("q1", "How many?", FAST, tracker)
    token = enter_question(scope)
    try:
        waiting = asyncio.create_task(wait_before_retry("agent", 1, "429 Too Many Requests"))
        await asyncio.sleep(0)
        [listed] = tracker.waiting()
        assert listed["question_id"] == "q1" and listed["stage"] == "agent"
        assert (listed["retry"], listed["max_retries"]) == (1, 2)
        await waiting
        assert tracker.waiting() == []
        await wait_before_retry("agent", 2, "429")
        with pytest.raises(RateLimitExhaustedError):
            await wait_before_retry("agent", 3, "429")
    finally:
        leave_question(token)
    assert scope.error_label().startswith("rate_limited: agent (still rate limited after 2 retries")
    assert scope.retries == {"agent": 2}


@pytest.mark.unit
@pytest.mark.asyncio
async def test_stopping_a_waiting_question_ends_the_wait_at_once():
    tracker = RateLimitTracker()
    scope = QuestionScope("q1", "How many?", RateLimitPolicy(3, wait_seconds=60), tracker)
    token = enter_question(scope)
    try:
        waiting = asyncio.create_task(wait_before_retry("llm_judge", 1, "429"))
        await asyncio.sleep(0)
        assert tracker.stop("q1") is True
        with pytest.raises(RateLimitStoppedError):
            await asyncio.wait_for(waiting, timeout=1)
    finally:
        leave_question(token)
    assert scope.error_label() == "rate_limited: llm_judge (retries stopped by the user)"
    assert tracker.stop("other") is False


# ── The agent call ──────────────────────────────────────────────────────────


def _runner(agent, dataset_provider, executor, reporter) -> EvaluationRunner:
    engine = AsyncMock(spec=EvaluationEngine)
    engine.run_all = AsyncMock(return_value=[])
    aggregator = MagicMock(spec=MetricsAggregator)
    aggregator.aggregate.return_value = MagicMock()
    runner = EvaluationRunner(
        dataset_provider=dataset_provider,
        agent_client=agent,
        query_executor=executor,
        reporter=reporter,
        evaluation_engine=engine,
        metrics_aggregator=aggregator,
        rate_limit_policy=FAST,
    )
    return runner


def _context(runner) -> object:
    return runner._engine.run_all.call_args[0][0]


RATE_LIMITED = AgentResponse(
    thread_id="t", status="error", error="agent tool error: Error code: 429 - rate limit exceeded"
)
ANSWER = AgentResponse(thread_id="t", status="completed", sql_query="SELECT 1")


@pytest.mark.unit
@pytest.mark.asyncio
async def test_rate_limited_agent_is_asked_again(mock_dataset_provider, mock_query_executor, mock_reporter):
    agent = AsyncMock()
    agent.run = AsyncMock(side_effect=[RATE_LIMITED, ANSWER])
    runner = _runner(agent, mock_dataset_provider, mock_query_executor, mock_reporter)

    await runner.run(dataset_name="d", allowed_tables=[], rate_limits=RateLimitTracker())

    ctx = _context(runner)
    assert agent.run.await_count == 2
    assert ctx.generated_sql == "SELECT 1" and ctx.error is None
    assert ctx.metadata["rate_limit_retries"] == {"agent": 1}


@pytest.mark.unit
@pytest.mark.asyncio
async def test_agent_crash_with_429_is_retried_too(mock_dataset_provider, mock_query_executor, mock_reporter):
    agent = AsyncMock()
    agent.run = AsyncMock(side_effect=[RuntimeError("HTTP 429 Too Many Requests"), ANSWER])
    runner = _runner(agent, mock_dataset_provider, mock_query_executor, mock_reporter)

    await runner.run(dataset_name="d", allowed_tables=[])

    ctx = _context(runner)
    assert ctx.generated_sql == "SELECT 1" and not ctx.agent_crashed and ctx.error is None


@pytest.mark.unit
@pytest.mark.asyncio
async def test_agent_still_rate_limited_marks_the_question(mock_dataset_provider, mock_query_executor, mock_reporter):
    agent = AsyncMock()
    agent.run = AsyncMock(return_value=RATE_LIMITED)
    runner = _runner(agent, mock_dataset_provider, mock_query_executor, mock_reporter)

    await runner.run(dataset_name="d", allowed_tables=[], rate_limits=RateLimitTracker())

    ctx = _context(runner)
    assert agent.run.await_count == 3  # the first call and 2 retries
    assert ctx.error.startswith("rate_limited: agent (still rate limited after 2 retries")


@pytest.mark.unit
@pytest.mark.asyncio
async def test_other_agent_errors_are_not_retried(mock_dataset_provider, mock_query_executor, mock_reporter):
    agent = AsyncMock()
    agent.run = AsyncMock(
        return_value=AgentResponse(thread_id="t", status="error", error="table not found")
    )
    runner = _runner(agent, mock_dataset_provider, mock_query_executor, mock_reporter)

    await runner.run(dataset_name="d", allowed_tables=[])

    assert agent.run.await_count == 1
    assert _context(runner).error == "table not found"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_time_shift_dates_reach_the_question(mock_dataset_provider, mock_query_executor, mock_reporter):
    agent = AsyncMock()
    agent.run = AsyncMock(return_value=ANSWER)
    runner = _runner(agent, mock_dataset_provider, mock_query_executor, mock_reporter)

    await runner.run(dataset_name="d", allowed_tables=[], time_shift_offsets_days=[-7, -1146])

    assert _context(runner).time_shift_offsets_days == [-7, -1146]


# ── The judges ──────────────────────────────────────────────────────────────


def _reply(content: str) -> httpx.Response:
    return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})


@pytest.mark.unit
@pytest.mark.asyncio
@respx.mock
async def test_order_judge_waits_out_a_rate_limit():
    route = respx.post("https://llm.example/v1/chat/completions").mock(
        side_effect=[
            httpx.Response(429, json={"error": "Too Many Requests"}),
            httpx.Response(429, json={"error": "Too Many Requests"}),
            _reply('{"requires_order": true, "reason": "top 5"}'),
        ]
    )
    judge = LlmOrderJudge("https://llm.example/v1", "k", "m")
    scope = QuestionScope("q1", "Top 5?", FAST, RateLimitTracker())
    token = enter_question(scope)
    try:
        decision = await judge.requires_order("Top 5 customers?")
    finally:
        leave_question(token)

    assert route.call_count == 3
    assert decision.requires_order is True
    assert scope.retries == {"order_judge": 2} and scope.gave_up is None


@pytest.mark.unit
@pytest.mark.asyncio
@respx.mock
async def test_judge_out_of_retries_marks_the_question():
    respx.post("https://llm.example/v1/chat/completions").mock(
        return_value=httpx.Response(429, json={"error": "Too Many Requests"})
    )
    judge = LlmOrderJudge("https://llm.example/v1", "k", "m")
    scope = QuestionScope("q1", "Top 5?", FAST, RateLimitTracker())
    token = enter_question(scope)
    try:
        decision = await judge.requires_order("Top 5 customers?")
    finally:
        leave_question(token)

    assert decision.source == "unavailable"
    assert scope.error_label().startswith("rate_limited: order_judge")


# ── The job API ─────────────────────────────────────────────────────────────


@pytest.mark.unit
@pytest.mark.asyncio
async def test_job_lists_waiting_questions_and_stops_one():
    from src.api.routes.text_to_sql_evaluation_router import (
        _job_response,
        stop_question_retry,
    )

    store = EvaluationJobStore()
    outcome: dict = {}

    async def runner(progress):
        scope = QuestionScope("q1", "How many?", RateLimitPolicy(3, 60), progress.rate_limits)
        token = enter_question(scope)
        try:
            await wait_before_retry("agent", 1, "429")
        except RateLimitStoppedError:
            outcome["label"] = scope.error_label()
        finally:
            leave_question(token)

    job = store.submit(runner)
    await asyncio.sleep(0.01)
    [waiting] = _job_response(job).rate_limited
    assert waiting["question_id"] == "q1"

    response = await stop_question_retry(job.job_id, "q1", job_store=store)
    assert response.job_id == job.job_id
    for _ in range(10):
        await asyncio.sleep(0)
    assert outcome["label"] == "rate_limited: agent (retries stopped by the user)"

    from fastapi import HTTPException

    with pytest.raises(HTTPException) as missing:
        await stop_question_retry("nope", "q1", job_store=store)
    assert missing.value.status_code == 404

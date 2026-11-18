"""
test_llm_judge.py — The LLM judge's second look at failed questions.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.domain.entities.evaluation_result import EvaluationResult
from src.domain.entities.query_result import QueryResult
from src.domain.services.evaluation_runner import EvaluationRunner
from src.infrastructure.llm.llm_false_negative_judge import (
    LlmFalseNegativeJudge,
    format_result,
    parse_verdict,
)
from src.ports.false_negative_judge import JudgeVerdict

EXPECTED = QueryResult(
    success=True,
    rows=[["paris", 500], ["berlin", 0]],
    columns=["city", "n"],
    row_count=2,
)
GENERATED = QueryResult(
    success=True,
    rows=[["paris", 500], ["berlin", None]],
    columns=["city", "n"],
    row_count=2,
)


def _runner(verdict: JudgeVerdict | None) -> tuple[EvaluationRunner, AsyncMock | None]:
    judge = None
    if verdict is not None:
        judge = MagicMock()
        judge.judge = AsyncMock(return_value=verdict)
    runner = EvaluationRunner(
        dataset_provider=MagicMock(),
        agent_client=MagicMock(),
        query_executor=MagicMock(),
        reporter=MagicMock(),
        evaluation_engine=MagicMock(),
        metrics_aggregator=MagicMock(),
        false_negative_judge=judge,
        judge_min_confidence=80,
    )
    return runner, judge


def _context(generated=GENERATED):
    return SimpleNamespace(
        query="Passengers per city?",
        expected_result=EXPECTED,
        generated_result=generated,
    )


def _failed_contains():
    return EvaluationResult(
        "contains_accuracy", 0.0, False, details={"reason": "row_count_mismatch"}
    )


def _judge_result(results):
    return next((r for r in results if r.evaluator_name == "llm_judge"), None)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_confident_false_negative_turns_the_question_into_a_pass():
    runner, judge = _runner(
        JudgeVerdict(True, 90, "NULL and 0 mean the same here", model="m")
    )
    contains = _failed_contains()
    results = [contains]

    await runner._judge_failed_question(_context(), results)

    kwargs = judge.judge.await_args.kwargs
    assert kwargs["question"] == "Passengers per city?"
    assert kwargs["failure_reason"] == "row_count_mismatch"
    assert contains.passed and contains.score == 1.0
    assert contains.details["overturned_by_llm_judge"] is True
    assert contains.details["original_score"] == 0.0
    verdict = _judge_result(results)
    assert verdict.passed and verdict.details["overturned"] is True
    assert verdict.details["reasoning"] == "NULL and 0 mean the same here"


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "verdict",
    [JudgeVerdict(True, 70, "unsure"), JudgeVerdict(False, 95, "missing rows")],
)
async def test_unsure_or_wrong_keeps_the_failure(verdict):
    runner, _ = _runner(verdict)
    contains = _failed_contains()
    results = [contains]
    await runner._judge_failed_question(_context(), results)
    assert not contains.passed and contains.score == 0.0
    judged = _judge_result(results)
    assert (
        judged.applicable
        and not judged.passed
        and judged.details["overturned"] is False
    )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_passed_questions_are_not_judged():
    runner, judge = _runner(JudgeVerdict(True, 99))
    results = [EvaluationResult("contains_accuracy", 1.0, True)]
    await runner._judge_failed_question(_context(), results)
    judge.judge.assert_not_awaited()
    assert _judge_result(results) is None


@pytest.mark.unit
@pytest.mark.asyncio
async def test_without_a_generated_result_the_judge_is_not_applicable():
    runner, judge = _runner(JudgeVerdict(True, 99))
    results = [_failed_contains()]
    await runner._judge_failed_question(
        _context(QueryResult.failure("syntax error")), results
    )
    judge.judge.assert_not_awaited()
    assert _judge_result(results).details["reason"] == "generated_sql_did_not_run"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_judge_errors_are_recorded_not_raised():
    runner, _ = _runner(JudgeVerdict(None, error="judge unavailable: 503"))
    contains = _failed_contains()
    results = [contains]
    await runner._judge_failed_question(_context(), results)
    judged = _judge_result(results)
    assert not judged.applicable and judged.details["reason"] == "judge_unavailable"
    assert not contains.passed


@pytest.mark.unit
@pytest.mark.asyncio
async def test_no_judge_configured_changes_nothing():
    runner, _ = _runner(None)
    results = [_failed_contains()]
    await runner._judge_failed_question(_context(), results)
    assert len(results) == 1


@pytest.mark.unit
def test_parse_verdict_tolerates_strings_and_thinking():
    reply = '<think>hmm</think> Here: {"is_false_negative": "true", "confidence": "85", "reasoning": "ok"}'
    assert parse_verdict(reply) == {
        "is_false_negative": True,
        "confidence": 85,
        "reasoning": "ok",
        "key_differences": [],
    }
    assert parse_verdict("no json") is None


@pytest.mark.unit
def test_format_result_shows_the_first_rows():
    big = QueryResult(
        success=True, rows=[[i] for i in range(60)], columns=["n"], row_count=60
    )
    text = format_result(big)
    assert text.splitlines()[0] == "n"
    assert "... 10 more rows not shown" in text
    assert "NULL" in format_result(GENERATED)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_llm_judge_fills_the_langfuse_prompt(monkeypatch):
    prompt = MagicMock(version=3)
    prompt.compile = MagicMock(return_value="filled prompt")
    langfuse = MagicMock()
    langfuse.get_prompt = MagicMock(return_value=prompt)
    judge = LlmFalseNegativeJudge(
        langfuse, "judge-prompt", "http://llm/v1", "k", "gpt-x"
    )
    ask = AsyncMock(
        return_value={
            "is_false_negative": True,
            "confidence": 88,
            "reasoning": "r",
            "key_differences": [],
        }
    )
    monkeypatch.setattr(judge, "_ask", ask)

    verdict = await judge.judge(
        question="Q?",
        expected=EXPECTED,
        generated=GENERATED,
        failure_reason="row_count_mismatch",
        failure_details={"reason": "row_count_mismatch", "x": 1},
    )

    langfuse.get_prompt.assert_called_once_with(
        "judge-prompt", label="production", type="text"
    )
    variables = prompt.compile.call_args.kwargs
    assert variables["user_question"] == "Q?"
    assert variables["expected_row_count"] == "2"
    assert variables["generated_result_formatted"].startswith("city | n")
    assert variables["failure_analysis"] == "row count mismatch (x: 1)"
    ask.assert_awaited_once_with("filled prompt")
    assert verdict == JudgeVerdict(True, 88, "r", [], "gpt-x", "judge-prompt", 3)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_llm_judge_without_its_prompt_reports_why():
    langfuse = MagicMock()
    langfuse.get_prompt = MagicMock(side_effect=RuntimeError("not found"))
    judge = LlmFalseNegativeJudge(langfuse, "missing", "http://llm/v1", "k", "gpt-x")
    verdict = await judge.judge(
        question="Q?",
        expected=EXPECTED,
        generated=GENERATED,
        failure_reason="r",
        failure_details={},
    )
    assert verdict.is_false_negative is None and "not found" in verdict.error

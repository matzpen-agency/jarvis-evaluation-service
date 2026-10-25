"""
test_order_policy.py — When does row order count?

Order only matters when the golden SQL has ORDER BY without LIMIT, the rows
match, the order differs — and the question asks for an order.
"""

from __future__ import annotations

import pytest

from src.domain.entities.dataset_item import DatasetItem
from src.domain.entities.evaluation_context import EvaluationContext
from src.domain.entities.query_result import QueryResult
from src.domain.evaluators.contains_evaluator import ContainsEvaluator
from src.ports.order_judge import OrderDecision


class FakeJudge:
    def __init__(self, requires_order: bool | None, reason: str = "because") -> None:
        self.decision = OrderDecision(requires_order, reason, "llm")
        self.questions: list[str] = []

    async def requires_order(self, question: str) -> OrderDecision:
        self.questions.append(question)
        return self.decision


def _ctx(expected_sql: str, generated_rows, *, grading: str = "", question: str = "q") -> EvaluationContext:
    item = DatasetItem(
        id="i1", input={"query": question}, expected_output={"sql": expected_sql},
        metadata={"grading_method": grading} if grading else {},
    )
    ctx = EvaluationContext(
        dataset_item=item, run_id="r", query=question, expected_sql=expected_sql, allowed_tables=[]
    )
    ctx.generated_sql = "SELECT ..."
    ctx.expected_result = QueryResult(success=True, rows=[[1], [2], [3]], columns=["v"])
    ctx.generated_result = QueryResult(success=True, rows=generated_rows, columns=["v"])
    return ctx


SCRAMBLED = [[3], [1], [2]]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_no_order_by_ignores_order_without_asking():
    judge = FakeJudge(True)
    result = await ContainsEvaluator(order_judge=judge).evaluate(_ctx("SELECT v FROM t", SCRAMBLED))
    assert result.score == 1.0
    assert judge.questions == []


@pytest.mark.unit
@pytest.mark.asyncio
async def test_order_by_with_limit_lets_the_rows_decide():
    judge = FakeJudge(True)
    ctx = _ctx("SELECT v FROM t ORDER BY v LIMIT 3", SCRAMBLED)
    result = await ContainsEvaluator(order_judge=judge).evaluate(ctx)
    assert result.score == 1.0
    assert "LIMIT" in result.details["order_policy"]
    assert judge.questions == []


@pytest.mark.unit
@pytest.mark.asyncio
async def test_matching_order_never_asks_the_judge():
    judge = FakeJudge(True)
    ctx = _ctx("SELECT v FROM t ORDER BY v", [[1], [2], [3]])
    result = await ContainsEvaluator(order_judge=judge).evaluate(ctx)
    assert result.score == 1.0
    assert judge.questions == []


@pytest.mark.unit
@pytest.mark.asyncio
async def test_wrong_rows_fail_without_asking_the_judge():
    judge = FakeJudge(False)
    ctx = _ctx("SELECT v FROM t ORDER BY v", [[1], [2], [9]])
    result = await ContainsEvaluator(order_judge=judge).evaluate(ctx)
    assert result.score == 0.0
    assert judge.questions == []


@pytest.mark.unit
@pytest.mark.asyncio
async def test_order_not_requested_passes_with_the_reason():
    judge = FakeJudge(False, "asks for revenue per territory only")
    ctx = _ctx("SELECT v FROM t ORDER BY v", SCRAMBLED, question="Revenue by territory?")
    result = await ContainsEvaluator(order_judge=judge).evaluate(ctx)
    assert result.score == 1.0
    assert result.details["order_judgment"]["reason"] == "asks for revenue per territory only"
    assert judge.questions == ["Revenue by territory?"]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_order_requested_fails():
    ctx = _ctx("SELECT v FROM t ORDER BY v", SCRAMBLED, question="הצג מהנמוך לגבוה")
    result = await ContainsEvaluator(order_judge=FakeJudge(True)).evaluate(ctx)
    assert result.score == 0.0
    assert result.details["reason"] == "row_order_mismatch"


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("judge", [None, FakeJudge(None, "order judge unavailable: timeout")])
async def test_no_decision_passes_flagged_not_verified(judge):
    ctx = _ctx("SELECT v FROM t ORDER BY v", SCRAMBLED)
    result = await ContainsEvaluator(order_judge=judge).evaluate(ctx)
    assert result.score == 1.0
    assert result.details["order_not_verified"] is True


@pytest.mark.unit
@pytest.mark.asyncio
async def test_explicit_grading_overrides_the_judge():
    judge = FakeJudge(False)
    ctx = _ctx("SELECT v FROM t ORDER BY v", SCRAMBLED, grading="list")
    result = await ContainsEvaluator(order_judge=judge).evaluate(ctx)
    assert result.score == 0.0  # BIRD "list" means order matters
    assert judge.questions == []

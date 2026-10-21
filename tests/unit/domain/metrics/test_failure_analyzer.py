"""
test_failure_analyzer.py — Unit tests for FailureAnalyzer.
"""

from __future__ import annotations

import pytest

from src.domain.entities.agent_response import AgentResponse
from src.domain.entities.evaluation_context import EvaluationContext
from src.domain.entities.query_result import QueryResult
from src.domain.metrics.failure_analyzer import FailureAnalyzer


@pytest.fixture
def analyzer() -> FailureAnalyzer:
    return FailureAnalyzer()


@pytest.mark.unit
def test_empty_samples(analyzer):
    analysis = analyzer.calculate([])
    assert analysis.total_failures == 0
    assert analysis.failure_rate == 0.0
    assert len(analysis.categories) == 0


@pytest.mark.unit
def test_agent_crash(analyzer, sample_dataset_item):
    ctx = EvaluationContext(
        dataset_item=sample_dataset_item,
        run_id="run-1",
        query="q",
        expected_sql="SELECT 1",
        allowed_tables=[],
    )
    ctx.agent_crashed = True

    analysis = analyzer.calculate([ctx])
    assert analysis.total_failures == 1
    assert analysis.agent_crash_count == 1
    assert analysis.agent_crash_rate == 1.0
    assert analysis.failure_rate == 1.0
    assert any(c.category == "agent_crash" for c in analysis.categories)


@pytest.mark.unit
def test_timeout(analyzer, sample_dataset_item):
    ctx = EvaluationContext(
        dataset_item=sample_dataset_item,
        run_id="run-1",
        query="q",
        expected_sql="SELECT 1",
        allowed_tables=[],
    )
    ctx.timed_out = True

    analysis = analyzer.calculate([ctx])
    assert analysis.total_failures == 1
    assert analysis.timeout_count == 1
    assert analysis.timeout_rate == 1.0


@pytest.mark.unit
def test_sql_execution_failure(analyzer, sample_dataset_item):
    # Case where agent produced no SQL or AgentResponse is None
    ctx = EvaluationContext(
        dataset_item=sample_dataset_item,
        run_id="run-1",
        query="q",
        expected_sql="SELECT 1",
        allowed_tables=[],
    )
    # agent_response is None
    analysis = analyzer.calculate([ctx])
    assert analysis.total_failures == 1
    assert analysis.sql_execution_failure_count == 1

    # Case where AgentResponse failed
    ctx.agent_response = AgentResponse(thread_id="t", status="error", error="Boom")
    analysis = analyzer.calculate([ctx])
    assert analysis.total_failures == 1
    assert analysis.sql_execution_failure_count == 1


@pytest.mark.unit
def test_trino_failure(analyzer, sample_dataset_item, successful_agent_response):
    ctx = EvaluationContext(
        dataset_item=sample_dataset_item,
        run_id="run-1",
        query="q",
        expected_sql="SELECT 1",
        allowed_tables=[],
    )
    ctx.agent_response = successful_agent_response
    ctx.generated_sql = "SELECT 1"
    ctx.expected_result = QueryResult.failure("Trino broken")
    ctx.generated_result = QueryResult(success=True, rows=[], columns=[], row_count=0)

    analysis = analyzer.calculate([ctx])
    assert analysis.total_failures == 1
    assert analysis.trino_failure_count == 1
    assert analysis.trino_failure_rate == 1.0


@pytest.mark.unit
def test_validation_failure(analyzer, sample_context):
    # sample_context succeeds by default. Let's make it fail.
    # We can do this by having generated_result be different from expected_result.
    # Actually, FailureAnalyzer uses ctx.failed property.
    # ctx.failed is True if not ctx.succeeded.
    # But wait, FailureAnalyzer check is:
    # elif (ctx.expected_result and not ctx.expected_result.success) or (ctx.generated_result and not ctx.generated_result.success): trino_fail
    # elif ctx.failed: validation_fail
    # If the queries execute successfully but produce different outputs, it is a validation failure.
    # Let's verify: in sample_context, agent_response is successful, expected_result and generated_result are successful.
    # But wait, does ctx.succeeded check expected_result.success and generated_result.success? Yes:
    # def succeeded(self) -> bool:
    #     return (self.agent_response is not None and self.agent_response.succeeded
    #             and self.expected_result is not None and self.expected_result.success
    #             and self.generated_result is not None and self.generated_result.success)
    # Wait, does ctx.failed check if the evaluator scores are high? No! EvaluationContext is independent of evaluator outputs.
    # Wait! In that case, what triggers ctx.failed in the eyes of FailureAnalyzer?
    # Let's look at EvaluationContext.failed:
    # return not self.succeeded
    # If self.succeeded is False, then ctx.failed is True.
    # So if there are no agent crashes, no timeouts, no missing/failed agent responses, and no trino execution errors,
    # but ctx.failed is True (for example, if expected_result is None or generated_result is None), then it counts as a validation failure.
    # Let's construct a context where expected_result is None (or generated_result is None) but AgentResponse is successful.

    ctx = EvaluationContext(
        dataset_item=sample_context.dataset_item,
        run_id="run-1",
        query="q",
        expected_sql="SELECT 1",
        allowed_tables=[],
        agent_response=sample_context.agent_response,
        generated_sql="SELECT 1",
    )
    # expected_result is None, generated_result is None.
    # Because agent_response is successful, it skips:
    # 1. agent_crashed (False)
    # 2. timed_out (False)
    # 3. agent_response is None or not succeeded (False)
    # 4. expected/generated result failure (False, because they are None, and not success is checked:
    #    `(ctx.expected_result and not ctx.expected_result.success)`)
    # 5. So it hits `elif ctx.failed:` -> validation_fail
    analysis = analyzer.calculate([ctx])
    assert analysis.total_failures == 1
    assert analysis.validation_failure_count == 1
    assert analysis.validation_failure_rate == 1.0

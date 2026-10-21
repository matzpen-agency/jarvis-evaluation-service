"""
test_evaluation_runner_robustness.py — Every item finishes and is reported,
whatever fails; generated SQL is adapted to the connection's dialect.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.domain.entities.agent_response import AgentResponse
from src.domain.entities.dataset_item import DatasetItem
from src.domain.entities.query_result import QueryResult
from src.domain.evaluators.contains_evaluator import ContainsEvaluator
from src.domain.services.evaluation_engine import EvaluationEngine
from src.domain.services.evaluation_runner import EvaluationRunner
from src.domain.services.metrics_aggregator import MetricsAggregator
from src.ports.connection_metadata_provider import ConnectionMetadataProvider


class _Metadata(ConnectionMetadataProvider):
    async def get_dialect(self, connection_id):
        return "postgres"

    async def get_table_schema_map(self, connection_id=None):
        return {"dimcustomer": {"customerkey": "integer"}}


def _items(n: int) -> list[DatasetItem]:
    return [
        DatasetItem(
            id=f"q{i}",
            input={"query": f"question {i}"},
            expected_output={"sql": "SELECT customerkey FROM dimcustomer"},
        )
        for i in range(n)
    ]


def _runner(agent, executor, reporter, items, **kwargs) -> EvaluationRunner:
    provider = AsyncMock()
    provider.get_dataset = AsyncMock(return_value=items)
    return EvaluationRunner(
        dataset_provider=provider,
        agent_client=agent,
        query_executor=executor,
        reporter=reporter,
        evaluation_engine=EvaluationEngine([ContainsEvaluator()]),
        metrics_aggregator=MetricsAggregator(),
        metadata_provider=_Metadata(),
        **kwargs,
    )


def _agent(sql: str | None = "SELECT customerkey FROM dimcustomer") -> AsyncMock:
    agent = AsyncMock()
    agent.run = AsyncMock(
        return_value=AgentResponse(thread_id="t", status="completed", sql_query=sql)
    )
    return agent


def _executor(result: QueryResult | None = None) -> AsyncMock:
    executor = AsyncMock()
    executor.execute = AsyncMock(
        return_value=result or QueryResult(success=True, rows=[[1]], columns=["customerkey"])
    )
    return executor


@pytest.mark.unit
@pytest.mark.asyncio
async def test_every_item_is_reported_even_when_things_fail(mock_reporter):
    """One crashing agent call, one hanging evaluation: the run still completes."""
    agent = _agent()
    agent.run.side_effect = [
        RuntimeError("agent exploded"),
        AgentResponse(thread_id="t", status="completed", sql_query="SELECT 1"),
        AgentResponse(thread_id="t", status="completed", sql_query="SELECT customerkey FROM dimcustomer"),
    ]

    async def execute(sql, connection_id=None):
        if sql == "SELECT 1":
            await asyncio.sleep(5)  # hangs past the evaluation budget
        return QueryResult(success=True, rows=[[1]], columns=["customerkey"])

    executor = AsyncMock()
    executor.execute = AsyncMock(side_effect=execute)
    runner = _runner(agent, executor, mock_reporter, _items(3), evaluation_timeout=0.2)

    run = await runner.run(dataset_name="ds", allowed_tables=[], agent_kwargs={"connection_id": 6})

    assert run.total_cases == 3
    assert mock_reporter.report_sample.await_count == 3
    reported = {c.kwargs["dataset_item_id"] for c in mock_reporter.report_sample.await_args_list}
    assert reported == {"q0", "q1", "q2"}
    errors = {c.question_id: c.error for c in run.cases}
    assert errors["q0"] == "agent exploded"
    assert errors["q1"].startswith("evaluation_timed_out")
    assert run.cases[2].scores["contains_accuracy"] == 1.0


@pytest.mark.unit
@pytest.mark.asyncio
async def test_sql_runs_on_the_items_connection(mock_reporter):
    executor = _executor()
    runner = _runner(_agent(), executor, mock_reporter, _items(1))
    await runner.run(dataset_name="ds", allowed_tables=[], agent_kwargs={"connection_id": 6})
    assert executor.execute.await_count == 2
    assert all(c.kwargs["connection_id"] == 6 for c in executor.execute.await_args_list)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_generated_trino_sql_is_transpiled_for_postgres_connection(mock_reporter):
    trino_sql = 'SELECT customerkey FROM "postgres"."public"."dimcustomer"'
    calls: list[str] = []

    async def execute(sql, connection_id=None):
        calls.append(sql)
        if '"postgres".' in sql:
            return QueryResult.failure("cross-database references are not implemented")
        return QueryResult(success=True, rows=[[1]], columns=["customerkey"])

    executor = AsyncMock()
    executor.execute = AsyncMock(side_effect=execute)
    runner = _runner(_agent(trino_sql), executor, mock_reporter, _items(1))

    run = await runner.run(dataset_name="ds", allowed_tables=[], agent_kwargs={"connection_id": 6})

    ctx = mock_reporter.report_sample.await_args.kwargs["context"]
    assert ctx.generated_sql == trino_sql
    assert ctx.executed_generated_sql == 'SELECT customerkey FROM "public"."dimcustomer"'
    assert ctx.metadata["generated_sql_transpiled"] is True
    assert run.cases[0].scores["contains_accuracy"] == 1.0


@pytest.mark.unit
@pytest.mark.asyncio
async def test_broken_golden_sql_is_not_applicable_not_zero(mock_reporter):
    async def execute(sql, connection_id=None):
        return QueryResult.failure('relation "dimcustomer" does not exist')

    executor = AsyncMock()
    executor.execute = AsyncMock(side_effect=execute)
    runner = _runner(_agent(), executor, mock_reporter, _items(1))

    run = await runner.run(dataset_name="ds", allowed_tables=[], agent_kwargs={"connection_id": 6})

    assert run.cases[0].scores == {}
    assert run.cases[0].not_applicable == ["contains_accuracy"]
    assert run.cases[0].error.startswith("expected_sql_execution_failed")
    assert run.accuracy.evaluated_counts["contains_accuracy"] == 0


@pytest.mark.unit
@pytest.mark.asyncio
async def test_progress_callback_reports_each_item(mock_reporter):
    progress: list[tuple[int, int]] = []
    runner = _runner(_agent(), _executor(), mock_reporter, _items(3))
    await runner.run(
        dataset_name="ds",
        allowed_tables=[],
        agent_kwargs={"connection_id": 6},
        progress_callback=lambda done, total: progress.append((done, total)),
    )
    assert progress == [(1, 3), (2, 3), (3, 3)]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_reporter_failure_does_not_abort_the_run():
    reporter = MagicMock()
    reporter.create_dataset_run_trace = AsyncMock(return_value="trace")
    reporter.report_sample = AsyncMock(side_effect=RuntimeError("langfuse down"))
    reporter.report_run_summary = AsyncMock()
    reporter.flush = AsyncMock()

    runner = _runner(_agent(), _executor(), reporter, _items(2))
    run = await runner.run(dataset_name="ds", allowed_tables=[], agent_kwargs={"connection_id": 6})
    assert run.total_cases == 2


@pytest.mark.unit
@pytest.mark.asyncio
async def test_truncated_results_are_counted_server_side(mock_reporter):
    async def execute(sql, connection_id=None):
        if sql.startswith("SELECT COUNT(*) FROM ("):
            total = 5000 if "dimcustomer" in sql and "WHERE" not in sql else 1200
            return QueryResult(success=True, rows=[[str(total)]], columns=["count"])
        return QueryResult(success=True, rows=[[1]], columns=["customerkey"], truncated=True)

    executor = AsyncMock()
    executor.execute = AsyncMock(side_effect=execute)
    agent = _agent("SELECT customerkey FROM dimcustomer WHERE customerkey > 0")
    runner = _runner(agent, executor, mock_reporter, _items(1))

    run = await runner.run(dataset_name="ds", allowed_tables=[], agent_kwargs={"connection_id": 6})

    assert run.cases[0].scores["contains_accuracy"] == 0.0  # 5000 vs 1200 rows: definite fail
    ctx = mock_reporter.report_sample.await_args.kwargs["context"]
    assert (ctx.expected_result.total_row_count, ctx.generated_result.total_row_count) == (5000, 1200)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_agent_call_that_ignores_cancellation_cannot_hang_the_run(mock_reporter):
    """
    Regression: a stuck agent stream swallowed cancellation, so wait_for()
    never returned and the whole run froze. The deadline must hold anyway.
    """

    async def stuck_agent(**kwargs):
        # Ignores the first cancellation (the deadline's), gives up on the
        # next one so the test's event loop can still shut down
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            pass
        await asyncio.sleep(10)

    agent = AsyncMock()
    agent.run = AsyncMock(side_effect=stuck_agent)
    runner = _runner(agent, _executor(), mock_reporter, _items(2), agent_timeout=0.2)

    run = await asyncio.wait_for(
        runner.run(dataset_name="ds", allowed_tables=[], agent_kwargs={"connection_id": 6}),
        timeout=5,
    )

    assert run.total_cases == 2
    assert all(c.error == "agent_call_timed_out" for c in run.cases)
    assert mock_reporter.report_sample.await_count == 2


@pytest.mark.unit
@pytest.mark.asyncio
async def test_run_with_deadline_returns_result_and_abandons_stuck_work():
    from src.domain.services.deadline import run_with_deadline

    async def quick():
        return 42

    async def stubborn():
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            pass  # ignores the deadline's cancellation
        await asyncio.sleep(10)

    assert await run_with_deadline(quick(), 1, label="quick") == 42
    with pytest.raises(TimeoutError, match="stubborn exceeded"):
        await asyncio.wait_for(run_with_deadline(stubborn(), 0.1, label="stubborn"), timeout=2)

"""
test_agent_engine_routing.py — Generated SQL runs where the agent ran it.

For connections mapped to the agent's Trino, the agent's SQL runs there (in
the agent's dialect) while the golden SQL keeps running on the connection.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.domain.entities.evaluation_context import EvaluationContext
from src.domain.entities.query_result import QueryResult
from src.domain.evaluators.sql_comparison_utils import compare_results
from src.domain.evaluators.time_shift_evaluator import TimeShiftEvaluator
from src.domain.services.evaluation_engine import EvaluationEngine
from src.domain.services.evaluation_runner import EvaluationRunner
from src.infrastructure.trino.agent_trino_query_executor import (
    AgentTrinoQueryExecutor,
    parse_catalog_map,
)
from src.infrastructure.trino.trino_query_executor import TrinoQueryExecutor


def _rows(*values) -> QueryResult:
    return QueryResult(success=True, rows=[[v] for v in values], columns=["v"], row_count=len(values))


class FakeAgentExecutor:
    def __init__(self, connection_ids: set[int]) -> None:
        self._covered = connection_ids
        self.execute = AsyncMock(return_value=_rows(1))

    def covers(self, connection_id):
        return connection_id in self._covered


def _runner(dataset_provider, agent_client, query_executor, reporter, agent_executor):
    engine = AsyncMock(spec=EvaluationEngine)
    engine.run_all = AsyncMock(return_value=[])
    engine.evaluator_names = []
    return EvaluationRunner(
        dataset_provider=dataset_provider,
        agent_client=agent_client,
        query_executor=query_executor,
        reporter=reporter,
        evaluation_engine=engine,
        metrics_aggregator=MagicMock(),
        agent_executor=agent_executor,
    ), engine


@pytest.mark.unit
@pytest.mark.asyncio
async def test_generated_sql_runs_on_the_agent_trino_for_mapped_connections(
    mock_dataset_provider, mock_agent_client, mock_query_executor, mock_reporter
):
    agent_executor = FakeAgentExecutor({6})
    runner, engine = _runner(
        mock_dataset_provider, mock_agent_client, mock_query_executor, mock_reporter, agent_executor
    )

    await runner.run(dataset_name="d", allowed_tables=[], agent_kwargs={"connection_id": 6})

    ctx = engine.run_all.call_args[0][0]
    assert ctx.generated_on_agent_engine is True
    agent_executor.execute.assert_called_once()
    assert agent_executor.execute.call_args.args[0] == ctx.generated_sql
    # Golden SQL stays on the connection
    mock_query_executor.execute.assert_called_once()
    assert mock_query_executor.execute.call_args.args[0] == ctx.expected_sql
    assert ctx.metadata["evaluation_config"]["generated_sql_engine"] == "agent_trino"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_unmapped_connections_run_both_on_the_connection(
    mock_dataset_provider, mock_agent_client, mock_query_executor, mock_reporter
):
    agent_executor = FakeAgentExecutor({6})
    runner, engine = _runner(
        mock_dataset_provider, mock_agent_client, mock_query_executor, mock_reporter, agent_executor
    )

    await runner.run(dataset_name="d", allowed_tables=[], agent_kwargs={"connection_id": 7})

    assert engine.run_all.call_args[0][0].generated_on_agent_engine is False
    agent_executor.execute.assert_not_called()
    assert mock_query_executor.execute.call_count == 2


@pytest.mark.unit
@pytest.mark.asyncio
async def test_time_shift_runs_generated_sql_on_the_agent_trino_in_trino_dialect(sample_dataset_item):
    connection = AsyncMock()
    connection.execute = AsyncMock(return_value=_rows(10))
    agent = AsyncMock()
    agent.execute = AsyncMock(return_value=_rows(10))
    ctx = EvaluationContext(
        dataset_item=sample_dataset_item,
        run_id="r",
        query="q",
        expected_sql="SELECT order_date FROM orders",
        allowed_tables=[],
        connection_id=6,
        dialect="postgres",
        schema_map={"orders": {"order_date": "date"}},
        generated_on_agent_engine=True,
    )
    ctx.generated_sql = "SELECT order_date FROM orders"

    result = await TimeShiftEvaluator(
        query_executor=connection, offsets_days=[-7], agent_executor=agent
    ).evaluate(ctx)

    assert result.score == 1.0
    golden_sql = connection.execute.call_args.args[0]
    generated_sql = agent.execute.call_args.args[0]
    assert "__shifted_orders" in golden_sql and "__shifted_orders" in generated_sql
    assert "DATE_ADD('DAY', -7" in generated_sql  # Trino syntax
    assert golden_sql != generated_sql  # Postgres syntax for the golden side


@pytest.mark.unit
def test_parse_catalog_map():
    assert parse_catalog_map("6=adventureworks, 7 = sales,") == {6: "adventureworks", 7: "sales"}
    assert parse_catalog_map("") == {}
    with pytest.raises(ValueError):
        parse_catalog_map("adventureworks")


class _FakeCursor:
    description = [("v",)]

    def __init__(self, n: int) -> None:
        self._rows = [(i,) for i in range(n)]

    def execute(self, sql):
        pass

    def fetchmany(self, size):
        return self._rows[:size]

    def fetchall(self):
        return self._rows

    def close(self):
        pass


class _FakeConnection:
    def __init__(self, n: int) -> None:
        self.n = n

    def cursor(self):
        return _FakeCursor(self.n)

    def close(self):
        pass


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize(("n", "returned", "truncated"), [(5, 5, False), (8, 5, True)])
async def test_trino_row_cap_marks_truncation(n, returned, truncated):
    executor = TrinoQueryExecutor(connection_factory=lambda: _FakeConnection(n), max_rows=5)
    result = await executor.execute("SELECT v FROM t")
    assert (result.row_count, result.truncated) == (returned, truncated)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_agent_trino_uses_the_mapped_catalog():
    catalogs: list[str] = []

    def factory(catalog):
        catalogs.append(catalog)
        return _FakeConnection(1)

    executor = AgentTrinoQueryExecutor(factory, {6: "adventureworks"})
    assert executor.covers(6) and not executor.covers(7)
    assert (await executor.execute("SELECT 1", connection_id=6)).success
    assert catalogs == ["adventureworks"]
    assert not (await executor.execute("SELECT 1", connection_id=7)).success


@pytest.mark.unit
def test_postgres_money_strings_match_trino_numbers():
    golden = QueryResult(
        success=True,
        rows=[["North America", "$79,353,361.53"], ["Europe", "-$1,200.50"]],
        columns=["territory", "revenue"],
    )
    generated = QueryResult(
        success=True,
        rows=[["North America", 79353361.53], ["Europe", -1200.5]],
        columns=["territory_group", "revenue"],
    )
    score, _ = compare_results(golden, generated, allow_extra_columns=False)
    assert score == 1.0

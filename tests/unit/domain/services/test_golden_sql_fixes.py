"""
test_golden_sql_fixes.py — Golden SQL that can't be judged as written.

  - Golden SQL that treats Postgres money as a number is retried with money
    columns cast to numeric.
  - A golden result with no data (no rows, or only NULLs) makes the
    result-based scores not applicable.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.domain.entities.evaluation_context import EvaluationContext
from src.domain.entities.query_result import QueryResult
from src.domain.evaluators.contains_evaluator import ContainsEvaluator
from src.domain.evaluators.execution_accuracy_evaluator import (
    ExecutionAccuracyEvaluator,
)
from src.domain.evaluators.sql_comparison_utils import (
    cast_money_to_numeric,
    has_no_data,
)
from src.domain.evaluators.time_shift_evaluator import TimeShiftEvaluator
from src.domain.services.evaluation_engine import EvaluationEngine
from src.domain.services.evaluation_runner import EvaluationRunner

SCHEMA = {
    "factinternetsales": {"orderdatekey": "integer", "salesamount": "money", "totalproductcost": "money"},
    "dimdate": {"datekey": "integer", "calendaryear": "smallint"},
}

AVG_MONEY = (
    "SELECT AVG(order_total) AS aov FROM (SELECT salesordernumber, SUM(salesamount) AS order_total "
    "FROM factinternetsales GROUP BY salesordernumber) orders"
)


# ── cast_money_to_numeric ──────────────────────────────────────────────────


@pytest.mark.unit
def test_money_columns_are_cast_and_aliases_kept():
    sql = "SELECT SUM(f.salesamount) FROM factinternetsales f JOIN dimdate d ON f.orderdatekey = d.datekey"
    out = cast_money_to_numeric(sql, SCHEMA, "postgres")
    assert "CAST(salesamount AS DECIMAL)" in out or "CAST(salesamount AS NUMERIC)" in out
    assert "CAST(totalproductcost AS" in out
    assert "AS f" in out
    assert "__numeric_dimdate" not in out  # no money columns: left alone


@pytest.mark.unit
def test_tables_inside_subqueries_are_cast():
    out = cast_money_to_numeric(AVG_MONEY, SCHEMA, "postgres")
    assert out.startswith("WITH __numeric_factinternetsales_1 AS")
    assert "FROM __numeric_factinternetsales_1 AS factinternetsales" in out


@pytest.mark.unit
def test_sql_without_money_is_unchanged():
    sql = "SELECT calendaryear FROM dimdate"
    assert cast_money_to_numeric(sql, SCHEMA, "postgres") == sql
    assert cast_money_to_numeric(AVG_MONEY, {}, "postgres") == AVG_MONEY


# ── runner retry ──────────────────────────────────────────────────────────


def _runner(sample_dataset_item, agent_client, reporter, executor):
    engine = AsyncMock(spec=EvaluationEngine)
    engine.run_all = AsyncMock(return_value=[])
    engine.evaluator_names = []
    provider = AsyncMock()
    provider.get_dataset = AsyncMock(return_value=[sample_dataset_item])
    metadata = AsyncMock()
    metadata.get_dialect = AsyncMock(return_value="postgres")
    metadata.get_table_schema_map = AsyncMock(return_value=SCHEMA)
    metadata.get_connection_name = AsyncMock(return_value="AdventureWorksDW")
    runner = EvaluationRunner(
        dataset_provider=provider,
        agent_client=agent_client,
        query_executor=executor,
        reporter=reporter,
        evaluation_engine=engine,
        metrics_aggregator=MagicMock(),
        metadata_provider=metadata,
    )
    return runner, engine


@pytest.mark.unit
@pytest.mark.asyncio
async def test_golden_sql_rejected_for_money_is_retried_cast_to_numeric(
    sample_dataset_item, mock_agent_client, mock_reporter
):
    sample_dataset_item.expected_output = {"sql": AVG_MONEY}
    ok = QueryResult(success=True, rows=[[1234.5]], columns=["aov"], row_count=1)

    async def execute(sql, connection_id=None):
        if sql == AVG_MONEY:
            return QueryResult.failure("Query failed: function avg(money) does not exist")
        return ok

    executor = AsyncMock()
    executor.execute = AsyncMock(side_effect=execute)
    runner, engine = _runner(sample_dataset_item, mock_agent_client, mock_reporter, executor)

    await runner.run(dataset_name="d", allowed_tables=[], agent_kwargs={"connection_id": 6})

    ctx = engine.run_all.call_args[0][0]
    assert ctx.expected_result is ok
    assert ctx.expected_sql == AVG_MONEY  # SQL-text metrics still see the original
    assert "CAST(salesamount AS" in ctx.executed_expected_sql
    assert ctx.metadata["expected_sql_money_cast"] is True
    assert "avg(money)" in ctx.metadata["expected_sql_raw_error"]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_other_golden_errors_are_not_retried(
    sample_dataset_item, mock_agent_client, mock_reporter
):
    sample_dataset_item.expected_output = {"sql": AVG_MONEY}
    executor = AsyncMock()
    executor.execute = AsyncMock(return_value=QueryResult.failure("relation \"x\" does not exist"))
    runner, engine = _runner(sample_dataset_item, mock_agent_client, mock_reporter, executor)

    await runner.run(dataset_name="d", allowed_tables=[], agent_kwargs={"connection_id": 6})

    ctx = engine.run_all.call_args[0][0]
    assert ctx.expected_result.success is False
    assert ctx.executed_expected_sql == AVG_MONEY
    golden_calls = [c for c in executor.execute.call_args_list if "salesamount" in c.args[0]]
    assert len(golden_calls) == 1


# ── golden result with no data ────────────────────────────────────────────


def _result(rows, truncated=False):
    return QueryResult(success=True, rows=rows, columns=["v"] * (len(rows[0]) if rows else 1),
                       row_count=len(rows), truncated=truncated)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("rows", "truncated", "expected"),
    [([], False, True), ([[None]], False, True), ([[None, None]], False, True),
     ([[0]], False, False), ([["x"]], False, False), ([], True, False)],
)
def test_has_no_data(rows, truncated, expected):
    assert has_no_data(_result(rows, truncated)) is expected


def _ctx(sample_dataset_item, expected_rows, generated_rows) -> EvaluationContext:
    ctx = EvaluationContext(
        dataset_item=sample_dataset_item,
        run_id="r",
        query="What was revenue in Q4 2013?",
        expected_sql="SELECT SUM(salesamount) FROM factinternetsales",
        allowed_tables=[],
        connection_id=6,
        dialect="postgres",
        schema_map={"factinternetsales": {"orderdate": "date", "salesamount": "money"}},
    )
    ctx.generated_sql = "SELECT COALESCE(SUM(salesamount), 0) FROM factinternetsales"
    ctx.expected_result = _result(expected_rows)
    ctx.generated_result = _result(generated_rows)
    return ctx


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("expected_rows", "generated_rows"), [([[None]], [[0.0]]), ([], [])]
)
@pytest.mark.parametrize("evaluator_cls", [ContainsEvaluator, ExecutionAccuracyEvaluator])
async def test_golden_without_data_is_not_applicable(
    sample_dataset_item, evaluator_cls, expected_rows, generated_rows
):
    result = await evaluator_cls().evaluate(_ctx(sample_dataset_item, expected_rows, generated_rows))
    assert result.applicable is False
    assert result.details["reason"] == "golden_result_has_no_data"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_time_shift_skips_golden_without_data(sample_dataset_item):
    executor = AsyncMock()
    result = await TimeShiftEvaluator(query_executor=executor).evaluate(
        _ctx(sample_dataset_item, [[None]], [[0.0]])
    )
    assert result.applicable is False
    executor.execute.assert_not_called()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_golden_with_data_is_still_scored(sample_dataset_item):
    result = await ContainsEvaluator().evaluate(_ctx(sample_dataset_item, [[5.0]], [[0.0]]))
    assert result.applicable is True
    assert result.score == 0.0

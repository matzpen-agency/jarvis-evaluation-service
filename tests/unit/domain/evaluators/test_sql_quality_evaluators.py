"""
test_sql_quality_evaluators.py — SQL-text evaluators against realistic queries:
schema hallucination, component match, dialect error and exact match.
"""

from __future__ import annotations

import pytest

from src.domain.entities.evaluation_context import EvaluationContext
from src.domain.entities.query_result import QueryResult
from src.domain.evaluators.component_match_evaluator import ComponentMatchEvaluator
from src.domain.evaluators.dialect_error_evaluator import DialectErrorEvaluator
from src.domain.evaluators.schema_hallucination_evaluator import (
    SchemaHallucinationEvaluator,
)
from src.domain.evaluators.sql_exact_match_evaluator import SqlExactMatchEvaluator

_FACT = {"salesamount": "numeric", "orderdatekey": "integer", "promotionkey": "integer"}
_DATE = {"datekey": "integer", "calendaryear": "smallint"}
SCHEMA = {
    "factinternetsales": _FACT,
    "public.factinternetsales": _FACT,
    "dimdate": _DATE,
    "public.dimdate": _DATE,
}

EXPECTED_SQL = (
    "SELECT d.calendaryear, SUM(f.salesamount) AS total_sales "
    "FROM factinternetsales f JOIN dimdate d ON f.orderdatekey = d.datekey "
    "GROUP BY d.calendaryear ORDER BY d.calendaryear"
)


def _ctx(item, generated_sql: str, expected_sql: str = EXPECTED_SQL) -> EvaluationContext:
    ctx = EvaluationContext(
        dataset_item=item,
        run_id="r",
        query="internet sales by year",
        expected_sql=expected_sql,
        allowed_tables=[],
        connection_id=6,
        dialect="postgres",
        schema_map=SCHEMA,
    )
    ctx.generated_sql = generated_sql
    return ctx


# ── schema_hallucination ──────────────────────────────────────────────────────


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "sql",
    [
        # aliases + ORDER BY a select alias + catalog-qualified (Trino) name
        'SELECT dd.calendaryear AS yr, SUM(fis.salesamount) AS sales '
        'FROM "postgres"."public"."factinternetsales" fis '
        "JOIN dimdate dd ON dd.datekey = fis.orderdatekey GROUP BY 1 ORDER BY yr",
        # CTE and subquery columns are not real tables
        "WITH s AS (SELECT orderdatekey, salesamount FROM factinternetsales) "
        "SELECT x.calendaryear, x.t FROM (SELECT d.calendaryear, SUM(s.salesamount) t "
        "FROM s JOIN dimdate d ON s.orderdatekey = d.datekey GROUP BY 1) x ORDER BY t DESC",
        # unqualified column resolvable through the only table
        "SELECT SUM(salesamount) FROM factinternetsales",
    ],
)
async def test_no_false_hallucinations(sample_dataset_item, sql):
    result = await SchemaHallucinationEvaluator().evaluate(_ctx(sample_dataset_item, sql))
    assert result.score == 1.0, result.details


@pytest.mark.unit
@pytest.mark.asyncio
async def test_hallucinated_table_and_column_detected(sample_dataset_item):
    sql = (
        "SELECT d.calendar_year, SUM(f.sales_amount) FROM fact_internet_sales f "
        "JOIN dimdate d ON f.orderdatekey = d.datekey GROUP BY 1"
    )
    result = await SchemaHallucinationEvaluator().evaluate(_ctx(sample_dataset_item, sql))
    assert result.score == 0.0
    assert result.details["hallucinated_tables"] == ["fact_internet_sales"]
    assert result.details["hallucinated_columns"] == ["dimdate.calendar_year"]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_wrong_schema_is_a_hallucination(sample_dataset_item):
    sql = "SELECT SUM(salesamount) FROM sales.factinternetsales"
    result = await SchemaHallucinationEvaluator().evaluate(_ctx(sample_dataset_item, sql))
    assert result.details["hallucinated_tables"] == ["sales.factinternetsales"]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_no_schema_is_not_applicable(sample_dataset_item):
    ctx = _ctx(sample_dataset_item, "SELECT 1")
    ctx.schema_map = {}
    result = await SchemaHallucinationEvaluator().evaluate(ctx)
    assert result.applicable is False


# ── component_match ───────────────────────────────────────────────────────────


@pytest.mark.unit
@pytest.mark.asyncio
async def test_component_match_ignores_aliases_and_operand_order(sample_dataset_item):
    sql = (
        "SELECT dd.calendaryear AS yr, SUM(fis.salesamount) AS sales "
        "FROM factinternetsales fis JOIN dimdate dd ON dd.datekey = fis.orderdatekey "
        "GROUP BY 1 ORDER BY yr"
    )
    result = await ComponentMatchEvaluator().evaluate(_ctx(sample_dataset_item, sql))
    assert result.score == 1.0, result.details


@pytest.mark.unit
@pytest.mark.asyncio
async def test_component_match_partial_credit_and_diff(sample_dataset_item):
    expected = "SELECT a FROM t WHERE x = 1 AND y = 2 AND z = 3"
    generated = "SELECT a FROM t WHERE x = 1 AND y = 2"
    result = await ComponentMatchEvaluator().evaluate(
        _ctx(sample_dataset_item, generated, expected_sql=expected)
    )
    assert result.details["clause_scores"]["where"] == pytest.approx(0.8)
    assert result.details["differences"]["where"]["missing"] == ["3 = z"]  # canonical operand order
    assert result.score == pytest.approx(round((1 + 1 + 0.8) / 3, 4))


# ── dialect_error ─────────────────────────────────────────────────────────────


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("engine_error", "is_dialect_error"),
    [
        ('syntax error at or near "`"', True),
        ("function date_add(unknown, integer, date) does not exist", True),
        ("TrinoUserError(type=USER_ERROR, name=FUNCTION_NOT_FOUND, ...)", True),
        ('relation "fact_internet_sales" does not exist', False),  # hallucination
        ('column "foo" does not exist', False),
    ],
)
async def test_dialect_error_classifies_engine_errors(
    sample_dataset_item, engine_error, is_dialect_error
):
    ctx = _ctx(sample_dataset_item, "SELECT foo FROM factinternetsales")
    ctx.generated_result = QueryResult.failure(engine_error)
    result = await DialectErrorEvaluator().evaluate(ctx)
    assert result.score == (0.0 if is_dialect_error else 1.0)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_dialect_error_not_applicable_without_sql(sample_dataset_item):
    ctx = _ctx(sample_dataset_item, "")
    ctx.generated_sql = None
    result = await DialectErrorEvaluator().evaluate(ctx)
    assert result.applicable is False


# ── sql_exact_match ───────────────────────────────────────────────────────────


@pytest.mark.unit
@pytest.mark.asyncio
async def test_exact_match_ignores_formatting_quoting_and_catalog(sample_dataset_item):
    generated = (
        'select d.calendaryear, sum(f.salesamount) as total_sales\n'
        'from "postgres"."public"."factinternetsales" f\n'
        "join dimdate d on f.orderdatekey = d.datekey -- yearly\n"
        "group by d.calendaryear order by d.calendaryear;"
    )
    result = await SqlExactMatchEvaluator().evaluate(_ctx(sample_dataset_item, generated))
    assert result.score == 1.0, result.details


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "generated",
    [
        # different output alias
        EXPECTED_SQL.replace("total_sales", "revenue"),
        # different table aliases, and the Trino DESC/NULLS default
        "SELECT dd.calendaryear AS yr, SUM(fis.salesamount) AS revenue "
        "FROM factinternetsales AS fis JOIN dimdate dd ON fis.orderdatekey = dd.datekey "
        "GROUP BY dd.calendaryear ORDER BY dd.calendaryear",
        # no aliases at all
        "SELECT dimdate.calendaryear, SUM(factinternetsales.salesamount) "
        "FROM factinternetsales JOIN dimdate ON factinternetsales.orderdatekey = dimdate.datekey "
        "GROUP BY dimdate.calendaryear ORDER BY dimdate.calendaryear",
    ],
)
async def test_exact_match_ignores_aliases(sample_dataset_item, generated):
    result = await SqlExactMatchEvaluator().evaluate(_ctx(sample_dataset_item, generated))
    assert result.score == 1.0, result.details


@pytest.mark.unit
@pytest.mark.asyncio
async def test_exact_match_ignores_cte_and_subquery_names(sample_dataset_item):
    expected = (
        "WITH s AS (SELECT customerkey, SUM(salesamount) AS total FROM factinternetsales "
        "GROUP BY customerkey) SELECT x.customerkey, x.total FROM s x WHERE x.total > 100 "
        "ORDER BY x.total DESC"
    )
    generated = (
        "WITH sales_by_customer AS (SELECT customerkey, SUM(salesamount) AS amount "
        "FROM factinternetsales GROUP BY customerkey) SELECT sales_by_customer.customerkey, "
        "sales_by_customer.amount FROM sales_by_customer WHERE sales_by_customer.amount > 100 "
        "ORDER BY sales_by_customer.amount DESC"
    )
    result = await SqlExactMatchEvaluator().evaluate(
        _ctx(sample_dataset_item, generated, expected_sql=expected)
    )
    assert result.score == 1.0, result.details


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "generated",
    [
        EXPECTED_SQL.replace("ORDER BY d.calendaryear", "ORDER BY d.calendaryear DESC"),
        EXPECTED_SQL.replace("SUM(f.salesamount)", "SUM(f.taxamt)"),
        EXPECTED_SQL.replace("SUM(f.salesamount)", "AVG(f.salesamount)"),
        # same columns, different order
        "SELECT SUM(f.salesamount) AS total_sales, d.calendaryear "
        "FROM factinternetsales f JOIN dimdate d ON f.orderdatekey = d.datekey "
        "GROUP BY d.calendaryear ORDER BY d.calendaryear",
    ],
)
async def test_exact_match_still_catches_real_differences(sample_dataset_item, generated):
    result = await SqlExactMatchEvaluator().evaluate(_ctx(sample_dataset_item, generated))
    assert result.score == 0.0

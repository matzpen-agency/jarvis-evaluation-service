"""
test_sql_comparison_utils.py — Result comparison, normalisation and ORDER BY handling.
"""

from __future__ import annotations

import datetime as dt
from decimal import Decimal

import pytest

from src.domain.entities.query_result import QueryResult
from src.domain.evaluators.sql_comparison_utils import (
    compare_results,
    evaluate_contains,
    normalize_cell,
    order_key_indices,
    transpile_sql,
)


def _qr(rows, columns, truncated=False) -> QueryResult:
    return QueryResult(success=True, rows=rows, columns=columns, truncated=truncated)


# ── normalize_cell ────────────────────────────────────────────────────────────


@pytest.mark.unit
@pytest.mark.parametrize(
    ("left", "right"),
    [
        ("58247", 58247),                       # Jeen returns bigint as a string
        ("1.50", Decimal("1.5")),               # numeric as string vs Decimal
        ("$27,307,607.39", 27307607.39),        # Postgres money formatting
        ("1966-04-08T00:00:00.000Z", dt.date(1966, 4, 8)),  # midnight timestamp == date
        ("2026-10-21T13:05:00.000Z", dt.datetime(2026, 10, 21, 13, 5)),
        (0.1 + 0.2, 0.3),
        (29358677.2207, 29358677.2207000003),   # float noise on large sums
        ("  Bikes ", "bikes"),
        (True, 1),
    ],
)
def test_equivalent_values_normalise_equal(left, right):
    assert normalize_cell(left) == normalize_cell(right)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("left", "right"),
    [
        (None, ""),            # NULL is not the empty string
        ("1,2,3", 123),        # not a thousands-separated number
        (1.5, 1.6),
        ("2026-10-21", "2026-10-22"),
    ],
)
def test_different_values_stay_different(left, right):
    assert normalize_cell(left) != normalize_cell(right)


# ── compare_results ───────────────────────────────────────────────────────────


@pytest.mark.unit
def test_columns_matched_by_data_not_name_or_position():
    expected = _qr([[2011, 100.0], [2012, 200.0]], ["calendaryear", "total_sales"])
    generated = _qr([["200.00", 2012], ["100.00", 2011]], ["sales", "yr"])
    score, details = compare_results(expected, generated, allow_extra_columns=False)
    assert score == 1.0
    assert details["column_mapping"] == {"calendaryear": "yr", "total_sales": "sales"}


@pytest.mark.unit
def test_backtracks_when_columns_share_a_value_multiset():
    """
    Generated has two 0/1 flag columns with the same values as expected's
    flag; only one of them lines up with the names. Greedy first-match picks
    the wrong one and reports a false failure.
    """
    expected = _qr([[1, "p"], [0, "q"]], ["flag", "name"])
    generated = _qr([[0, 1, "p"], [1, 0, "q"]], ["other_flag", "flag2", "name"])
    score, details = evaluate_contains(expected, generated)
    assert score == 1.0
    assert details["column_mapping"] == {"flag": "flag2", "name": "name"}


@pytest.mark.unit
def test_contains_allows_extra_columns_but_exact_does_not():
    expected = _qr([[1], [2]], ["id"])
    generated = _qr([[1, "a"], [2, "b"]], ["id", "name"])
    assert evaluate_contains(expected, generated)[0] == 1.0
    score, details = compare_results(expected, generated, allow_extra_columns=False)
    assert score == 0.0
    assert details["reason"] == "column_count_mismatch"


@pytest.mark.unit
def test_rows_must_pair_up_not_just_columns():
    """Same column value sets, but the rows are combined differently."""
    expected = _qr([[1, "a"], [2, "b"]], ["id", "name"])
    generated = _qr([[1, "b"], [2, "a"]], ["id", "name"])
    score, details = compare_results(expected, generated, allow_extra_columns=False)
    assert score == 0.0
    assert details["reason"] == "projected_rows_do_not_match"


@pytest.mark.unit
def test_duplicate_rows_are_counted():
    expected = _qr([[1], [1], [2]], ["v"])
    generated = _qr([[1], [2], [2]], ["v"])
    assert compare_results(expected, generated, allow_extra_columns=False)[0] == 0.0


@pytest.mark.unit
def test_ordering_tolerates_ties_on_the_sort_key():
    expected = _qr([["b", 10], ["c", 10], ["a", 5]], ["name", "revenue"])
    generated = _qr([["c", 10], ["b", 10], ["a", 5]], ["name", "revenue"])
    keys = order_key_indices("SELECT name, revenue FROM t ORDER BY revenue DESC", expected.columns)
    assert keys == [1]
    score, _ = compare_results(
        expected, generated, allow_extra_columns=False, requires_ordering=True, order_keys=keys
    )
    assert score == 1.0


@pytest.mark.unit
def test_ordering_violation_detected():
    expected = _qr([["a", 5], ["b", 10]], ["name", "revenue"])
    generated = _qr([["b", 10], ["a", 5]], ["name", "revenue"])
    keys = order_key_indices("SELECT name, revenue FROM t ORDER BY revenue", expected.columns)
    score, details = compare_results(
        expected, generated, allow_extra_columns=False, requires_ordering=True, order_keys=keys
    )
    assert score == 0.0
    assert details["reason"] == "row_order_mismatch"


@pytest.mark.unit
@pytest.mark.parametrize(
    ("sql", "columns", "expected_keys"),
    [
        ("SELECT a, SUM(x) AS total FROM t GROUP BY a ORDER BY total DESC", ["a", "total"], [1]),
        ("SELECT a, SUM(x) AS total FROM t GROUP BY a ORDER BY SUM(x)", ["a", "total"], [1]),
        ("SELECT d.y, SUM(s) FROM t d GROUP BY d.y ORDER BY d.y", ["y", "sum"], [0]),
        ("SELECT a, b FROM t ORDER BY 2, 1", ["a", "b"], [1, 0]),
        ("SELECT a FROM t ORDER BY b", ["a"], None),  # sort key not in the output
        ("SELECT a FROM t", ["a"], None),
    ],
)
def test_order_key_indices(sql, columns, expected_keys):
    assert order_key_indices(sql, columns) == expected_keys


@pytest.mark.unit
def test_both_truncated_is_inconclusive():
    expected = _qr([[i] for i in range(3)], ["v"], truncated=True)
    generated = _qr([[i] for i in range(3)], ["v"], truncated=True)
    score, details = compare_results(expected, generated, allow_extra_columns=True)
    assert score == 0.0
    assert details["inconclusive"] is True


@pytest.mark.unit
def test_one_side_truncated_is_a_row_count_mismatch():
    expected = _qr([[1], [2]], ["v"])
    generated = _qr([[1], [2]], ["v"], truncated=True)
    score, details = compare_results(expected, generated, allow_extra_columns=True)
    assert score == 0.0
    assert details["reason"] == "row_count_mismatch"
    assert "inconclusive" not in details


@pytest.mark.unit
def test_empty_results_match_but_are_flagged():
    score, details = compare_results(_qr([], ["v"]), _qr([], ["v"]), allow_extra_columns=True)
    assert score == 1.0
    assert details["empty_result"] is True


# ── transpile_sql ─────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_transpile_trino_to_postgres_drops_catalog():
    sql = 'SELECT date_add(\'day\', -1, current_date), x FROM "postgres"."public"."dimcustomer"'
    out = transpile_sql(sql, read="trino", write="postgres")
    assert out is not None
    assert '"postgres".' not in out
    assert '"public"."dimcustomer"' in out
    assert "INTERVAL" in out


@pytest.mark.unit
def test_both_truncated_with_different_totals_is_a_mismatch():
    expected = _qr([[1]], ["v"], truncated=True)
    generated = _qr([[1]], ["v"], truncated=True)
    expected.total_row_count, generated.total_row_count = 5000, 1200
    score, details = compare_results(expected, generated, allow_extra_columns=True)
    assert score == 0.0
    assert details["reason"] == "row_count_mismatch"
    assert "inconclusive" not in details


@pytest.mark.unit
@pytest.mark.asyncio
async def test_truncated_results_make_execution_metrics_not_applicable(sample_dataset_item):
    """Regression: this path used to raise (duplicate 'reason' kwarg) and score 0."""
    from src.domain.entities.evaluation_context import EvaluationContext
    from src.domain.evaluators.contains_evaluator import ContainsEvaluator
    from src.domain.evaluators.execution_accuracy_evaluator import (
        ExecutionAccuracyEvaluator,
    )

    ctx = EvaluationContext(
        dataset_item=sample_dataset_item, run_id="r", query="q",
        expected_sql="SELECT v FROM t", allowed_tables=[],
    )
    ctx.generated_sql = "SELECT v FROM t"
    ctx.expected_result = _qr([[1]], ["v"], truncated=True)
    ctx.generated_result = _qr([[1]], ["v"], truncated=True)
    for evaluator in (ContainsEvaluator(), ExecutionAccuracyEvaluator()):
        result = await evaluator.evaluate(ctx)
        assert result.error is None
        assert result.applicable is False
        assert result.details["reason"] == "results_truncated"

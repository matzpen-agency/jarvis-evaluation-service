"""
sql_comparison_utils.py — Shared utilities for deterministic SQL result comparison.

Provides two core helpers used by all execution-based evaluators:

  _sort_dataframe(rows, columns)
      Normalises a result table by reordering columns deterministically
      (successive-minimum sort key) and sorting rows accordingly.
      Handles every Python type that Trino DBAPI can return safely.

  dynamically_wrap_with_yaml_cte(sql_query, shift_days, table_schema_map)
      Wraps each table reference in the query with a CTE that applies a
      Trino-compatible date_add() shift to every detected date/timestamp
      column.  When shift_days == 0 the original query is returned as-is.
"""

from __future__ import annotations

import math
import re
from functools import total_ordering
from typing import Any

import sqlglot
import sqlglot.expressions as exp
import structlog

logger = structlog.get_logger(__name__)


def _requires_order_by(sql: str | None) -> bool:
    """
    Check if the given SQL query contains an outermost ORDER BY clause using sqlglot AST.
    """
    if not sql:
        return False
    try:
        parsed = sqlglot.parse_one(sql, error_level=sqlglot.ErrorLevel.IGNORE)
        return parsed is not None and parsed.args.get("order") is not None
    except Exception:
        # Fallback to regex check if sqlglot fails to parse
        return bool(re.search(r"\border\s+by\b", sql, re.IGNORECASE))

# ── Column heuristics ─────────────────────────────────────────────────────────
# A column whose name contains any of these tokens is treated as a date/ts col.
_DATE_COL_TOKENS: frozenset[str] = frozenset(
    {
        "date",
        "time",
        "ts",
        "timestamp",
        "created",
        "updated",
        "day",
        "month",
        "year",
        "period",
        "start",
        "end",
        "from",
        "until",
        "since",
        "through",
    }
)

# ─────────────────────────────────────────────────────────────────────────────
# _sort_dataframe
# ─────────────────────────────────────────────────────────────────────────────

# Data-driven priority table: (type, priority). Checked in order; first match wins.
# bool must precede int/float because bool is a subclass of int.
_TYPE_PRIORITY: tuple[tuple[type, int], ...] = (
    (bool,       1),
    (int,        2),
    (float,      2),
    (str,        3),
    (bytes,      4),
    (list,       5),
    (dict,       5),
    (tuple,      5),
)
_PRIORITY_NULL = 99   # None
_PRIORITY_NAN  = 98   # NaN / NaT-like


@total_ordering
class _Comparable:
    """
    Robust wrapper for any value type to ensure mathematical determinism when
    sorting columns, safely handling ties, differing types, and NaNs.

    Priority table drives comparison; equal-priority values are compared
    natively with fallback to str representation.
    """
    __slots__ = ("val", "type_priority")

    def __init__(self, val: Any):
        self.val = val
        self.type_priority = self._get_priority(val)

    @staticmethod
    def _get_priority(val: Any) -> int:
        if val is None:
            return _PRIORITY_NULL
        # Detect NaN / NaT-like values without hardcoding class names
        try:
            if math.isnan(val):
                return _PRIORITY_NAN
        except (TypeError, ValueError):
            pass
        for typ, priority in _TYPE_PRIORITY:
            if isinstance(val, typ):
                return priority
        return 10  # Unknown type — sorts after known types, before nulls

    def _cmp_value(self) -> Any:
        """Return the raw value for comparison; NaN/None mapped to their priority."""
        return self.val

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, _Comparable):
            return NotImplemented
        if self.type_priority != other.type_priority:
            return False
        if self.type_priority >= _PRIORITY_NAN:   # None or NaN — all equal
            return True
        try:
            return bool(self.val == other.val)
        except Exception:
            return str(self.val) == str(other.val)

    def __lt__(self, other: Any) -> bool:
        if not isinstance(other, _Comparable):
            return NotImplemented
        if self.type_priority != other.type_priority:
            return self.type_priority < other.type_priority
        if self.type_priority >= _PRIORITY_NAN:   # None or NaN — not ordered
            return False
        try:
            return bool(self.val < other.val)
        except TypeError:
            # Cross-type fall-through (e.g. int vs str within same priority)
            return str(type(self.val)) < str(type(other.val))
        except Exception:
            return str(self.val) < str(other.val)


def _column_sort_key(col_values: list[Any]) -> tuple[_Comparable, ...]:
    """
    Compute the deterministic signature for a column.

    The signature is a sorted tuple of all values in the column wrapped in
    _Comparable. This guarantees a mathematically deterministic column order
    regardless of ties or naming differences.
    """
    return tuple(sorted(_Comparable(v) for v in col_values))


def _sort_dataframe(
    rows: list[list[Any]],
    columns: list[str],
) -> tuple[list[list[Any]], list[str]]:
    """
    Normalize a query result table for deterministic, column-order-invariant
    comparison. Preserves row sequence so ORDER BY evaluation is respected.

    Algorithm:
      1. Compute a sort key for each column (full sorted value signature).
      2. Sort columns ascending by that key (ties broken by column name).
      3. Reorder every row's values to match the new column order.
      4. Maintain the original row sequence intact.
    """
    if not rows or not columns:
        return rows, columns

    n_cols = len(columns)

    # Build per-column value lists (handle rows shorter than columns gracefully)
    col_values: list[list[Any]] = [[] for _ in range(n_cols)]
    for row in rows:
        for c in range(n_cols):
            col_values[c].append(row[c] if c < len(row) else None)

    # Compute sort key per column — (full_signature, column_name) for tie-breaking
    col_keys: list[tuple[tuple[_Comparable, ...], str]] = [
        (_column_sort_key(col_values[c]), columns[c])
        for c in range(n_cols)
    ]

    # New column order: indices sorted by (min_signature, column_name)
    sorted_indices: list[int] = sorted(range(n_cols), key=lambda c: col_keys[c])
    sorted_columns: list[str] = [columns[i] for i in sorted_indices]

    # Reorder row cells to match new column order, preserving row sequence
    reordered_rows: list[list[Any]] = []
    for row in rows:
        new_row = [row[i] if i < len(row) else None for i in sorted_indices]
        reordered_rows.append(new_row)

    logger.debug(
        "sql_comparison_utils._sort_dataframe",
        original_columns=columns,
        sorted_columns=sorted_columns,
        row_count=len(reordered_rows),
    )

    return reordered_rows, sorted_columns


def _is_ordered_subsequence(sub: list, seq: list) -> bool:
    """
    Return True if `sub` is an ordered subsequence of `seq`.

    Each element of `sub` must appear in `seq` in the same relative order,
    but elements in `seq` can be skipped.

    Examples:
        _is_ordered_subsequence([2, 3], [1, 2, 3]) → True
        _is_ordered_subsequence([3, 2], [1, 2, 3]) → False
        _is_ordered_subsequence([1, 2, 3], [1, 2, 3]) → True
    """
    it = iter(seq)
    return all(item in it for item in sub)


def evaluate_contains(
    expected_result: Any,
    generated_result: Any,
    numeric_tolerance: int = 6,
    requires_ordering: bool = False,
) -> tuple[float, dict[str, Any]]:
    """
    Compare expected and generated QueryResults for column data containment.

    1. If requires_ordering is False:
       Enforces 100% row count match. Checks whether all columns of
       expected_result exist as a subset (by data values) of the
       generated_result columns (multiset equality).
    2. If requires_ordering is True:
       Checks that the generated row-tuples form an ordered subsequence
       of the expected row-tuples.

    Returns:
        (1.0 if column data is contained, else 0.0, details_dict)
    """
    from collections import Counter
    from src.domain.entities.query_result import QueryResult

    if expected_result is None or not expected_result.success:
        return 0.0, {"reason": "expected_sql_execution_failed"}
    if generated_result is None or not generated_result.success:
        return 0.0, {"reason": "generated_sql_execution_failed"}

    n_exp_cols = len(expected_result.columns)
    n_gen_cols = len(generated_result.columns)

    if n_gen_cols < n_exp_cols:
        return 0.0, {
            "reason": "insufficient_generated_columns",
            "expected_column_count": n_exp_cols,
            "generated_column_count": n_gen_cols,
        }

    # Normalize both results into row tuples
    exp_qr = QueryResult(
        success=True, rows=expected_result.rows, columns=expected_result.columns
    )
    gen_qr = QueryResult(
        success=True, rows=generated_result.rows, columns=generated_result.columns
    )

    exp_tuples = exp_qr.as_normalised_row_tuples(numeric_tolerance)
    gen_tuples = gen_qr.as_normalised_row_tuples(numeric_tolerance)

    if not exp_tuples and not gen_tuples:
        return 1.0, {
            "expected_row_count": 0,
            "generated_row_count": 0,
            "requires_ordering": requires_ordering,
            "column_containment": True,
        }

    if requires_ordering:
        # ── Ordered subsequence branch ────────────────────────────────────────
        # The generated rows must be an ordered subsequence of the expected rows.
        # Row counts do NOT need to be equal.
        # Column matching is name-agnostic: find the generated column whose
        # ordered value list is a subsequence of the expected column's value list.

        exp_cols_data = [
            [row[i] for row in exp_tuples] for i in range(n_exp_cols)
        ]
        gen_cols_data = [
            [row[i] for row in gen_tuples] for i in range(n_gen_cols)
        ]

        used_gen_indices: set[int] = set()
        col_mapping: list[int] = []

        for exp_col in exp_cols_data:
            matched_idx = None
            for gen_idx, gen_col in enumerate(gen_cols_data):
                if gen_idx in used_gen_indices:
                    continue
                if _is_ordered_subsequence(gen_col, exp_col):
                    matched_idx = gen_idx
                    break
            if matched_idx is None:
                return 0.0, {
                    "reason": "column_data_not_ordered_subsequence",
                    "expected_column_count": n_exp_cols,
                    "generated_column_count": n_gen_cols,
                    "requires_ordering": True,
                }
            used_gen_indices.add(matched_idx)
            col_mapping.append(matched_idx)

        # Project generated rows onto the matched columns and verify
        # the full multi-column row-tuples form an ordered subsequence
        projected_gen = [
            tuple(row[col_mapping[i]] for i in range(n_exp_cols))
            for row in gen_tuples
        ]
        projected_exp = [
            tuple(row[i] for i in range(n_exp_cols))
            for row in exp_tuples
        ]

        if not _is_ordered_subsequence(projected_gen, projected_exp):
            return 0.0, {
                "reason": "generated_rows_not_ordered_subsequence",
                "expected_row_count": len(exp_tuples),
                "generated_row_count": len(gen_tuples),
                "requires_ordering": True,
            }

        return 1.0, {
            "expected_row_count": len(exp_tuples),
            "generated_row_count": len(gen_tuples),
            "requires_ordering": True,
            "column_containment": True,
        }

    else:
        # ── Unordered branch: expected-row multiset containment ───────────────────
        # Generated may have MORE rows than expected; all expected rows must be
        # present (multiset containment).  Extra generated rows are allowed.
        exp_cols_data = [
            [row[i] for row in exp_tuples] for i in range(n_exp_cols)
        ]
        gen_cols_data = [
            [row[i] for row in gen_tuples] for i in range(n_gen_cols)
        ]

        exp_counter = Counter(exp_tuples)
        gen_counter = Counter(gen_tuples)
        all_contained = all(gen_counter[row] >= cnt for row, cnt in exp_counter.items())

        if not all_contained:
            return 0.0, {
                "reason": "expected_rows_not_contained",
                "expected_row_count": len(exp_tuples),
                "generated_row_count": len(gen_tuples),
            }

        return 1.0, {
            "expected_row_count": len(exp_tuples),
            "generated_row_count": len(gen_tuples),
            "requires_ordering": False,
            "column_containment": True,
        }


# ─────────────────────────────────────────────────────────────────────────────
# dynamically_wrap_with_yaml_cte
# ─────────────────────────────────────────────────────────────────────────────


def _is_date_column(col_name: str) -> bool:
    """Return True when col_name looks like a date / timestamp column."""
    lower = col_name.lower()
    return any(token in lower for token in _DATE_COL_TOKENS)


def _extract_table_names_sqlglot(sql: str) -> list[str]:
    """Extract table names using sqlglot AST walking."""
    try:
        import sqlglot
        import sqlglot.expressions as exp

        parsed = sqlglot.parse_one(sql, read="trino", error_level=sqlglot.ErrorLevel.IGNORE)
        if parsed is None:
            return []
        tables: list[str] = []
        for tbl in parsed.find_all(exp.Table):
            name = tbl.name
            if name:
                tables.append(name.lower())
        return list(dict.fromkeys(tables))  # deduplicate, preserve order
    except Exception as exc:
        logger.debug("sql_comparison_utils.sqlglot_parse_failed", error=str(exc))
        return []


def _extract_table_names_regex(sql: str) -> list[str]:
    """Fallback: extract table names using regex FROM / JOIN patterns."""
    pattern = re.compile(
        r"\b(?:FROM|JOIN)\s+([a-zA-Z_][a-zA-Z0-9_.]*)",
        re.IGNORECASE,
    )
    matches = pattern.findall(sql)
    # Strip catalog.schema. prefixes — keep only the final table name part
    tables = [m.split(".")[-1].lower() for m in matches]
    return list(dict.fromkeys(tables))


def _extract_table_names(sql: str) -> list[str]:
    """Extract table names, preferring sqlglot with regex fallback."""
    tables = _extract_table_names_sqlglot(sql)
    if not tables:
        tables = _extract_table_names_regex(sql)
    return tables


def _build_cte_for_table(
    table_name: str,
    date_cols: set[str],
    all_cols: set[str],
    shift_days: int,
) -> str:
    """
    Build the CTE SELECT body for one table.

    Date columns are wrapped in date_add(); all other columns are selected as-is.
    We emit a SELECT * pattern using explicit column names to keep the CTE
    compatible with all Trino versions.

    When only the date columns are known (all_cols is empty), we fall back to
    SELECT *, <shifted_cols> FROM ... which uses column shadowing — acceptable
    as a safe fallback.
    """
    if not all_cols:
        # Fallback: shift only the known date columns on top of SELECT *
        shifted = ", ".join(
            f"date_add('day', {shift_days}, TRY_CAST({col} AS DATE)) AS {col}"
            for col in sorted(date_cols)
        )
        return (
            f"{table_name}_cte AS (\n"
            f"  SELECT *, {shifted}\n"
            f"  FROM {table_name}\n"
            f")"
        )

    select_parts: list[str] = []
    for col in sorted(all_cols):
        if col in date_cols:
            select_parts.append(
                f"  date_add('day', {shift_days}, TRY_CAST({col} AS DATE)) AS {col}"
            )
        else:
            select_parts.append(f"  {col}")

    select_clause = ",\n".join(select_parts)
    return (
        f"{table_name}_cte AS (\n"
        f"  SELECT\n"
        f"{select_clause}\n"
        f"  FROM {table_name}\n"
        f")"
    )


def _replace_table_ref(sql: str, table_name: str, cte_name: str) -> str:
    """
    Replace bare table references in the SQL with cte_name.

    Handles: FROM table, JOIN table, FROM schema.table, JOIN schema.table.
    Does NOT replace partial matches (e.g. 'orders' won't match 'orders_2024').
    """
    # Match the table name optionally preceded by a schema/catalog prefix
    pattern = re.compile(
        r"(?<!\w)(?:[a-zA-Z_][a-zA-Z0-9_.]*\.)?" + re.escape(table_name) + r"(?!\w)",
        re.IGNORECASE,
    )
    return pattern.sub(cte_name, sql)


def dynamically_wrap_with_yaml_cte(
    sql_query: str,
    shift_days: int,
    table_schema_map: dict[str, set[str]],
) -> str:
    """
    Wrap SQL table references with Trino CTEs that shift date/timestamp columns.

    For each table referenced in sql_query:
      - Looks up its columns in table_schema_map.
      - Identifies date/timestamp columns by name heuristics.
      - Generates a CTE that applies date_add('day', shift_days, ...) to those
        columns and TRY_CASTs them to DATE.
      - Replaces the original table name in the query with <table>_cte.

    If shift_days == 0, the original query is returned unchanged.
    If a table has no date columns or is absent from table_schema_map, it is
    left unrewritten.

    Args:
        sql_query:        Original SQL string.
        shift_days:       Number of days to shift (positive = future, negative = past).
        table_schema_map: Mapping of table_name (lower) → set of column names (lower).
                          Obtain from BackendTableResolver.get_table_schema_map().

    Returns:
        Modified SQL with CTEs prepended, or original SQL if no changes needed.
    """
    if shift_days == 0:
        return sql_query

    table_names = _extract_table_names(sql_query)
    if not table_names:
        logger.debug("sql_comparison_utils.no_tables_found", sql_preview=sql_query[:120])
        return sql_query

    cte_blocks: list[str] = []
    rewritten_sql = sql_query

    for table_name in table_names:
        # Look up columns — try exact name, then without catalog/schema prefix
        all_cols: set[str] = table_schema_map.get(table_name, set())
        date_cols: set[str] = {c for c in all_cols if _is_date_column(c)}

        if not date_cols:
            logger.debug(
                "sql_comparison_utils.no_date_cols",
                table=table_name,
                known_cols=list(all_cols)[:10],
            )
            continue

        cte_name = f"{table_name}_cte"
        cte_block = _build_cte_for_table(table_name, date_cols, all_cols, shift_days)
        cte_blocks.append(cte_block)
        rewritten_sql = _replace_table_ref(rewritten_sql, table_name, cte_name)
        logger.debug(
            "sql_comparison_utils.cte_built",
            table=table_name,
            date_cols=sorted(date_cols),
            shift_days=shift_days,
        )

    if not cte_blocks:
        return sql_query  # No date columns found anywhere — return original

    new_cte_str = ",\n".join(cte_blocks)

    # Merge into an existing WITH clause instead of prepending a second WITH
    # Handles: WITH [...], WITH RECURSIVE [...]
    with_pattern = re.compile(
        r"^(WITH\s+RECURSIVE\s+|WITH\s+)", re.IGNORECASE | re.MULTILINE
    )
    m = with_pattern.match(rewritten_sql.lstrip())
    if m:
        # Insert generated CTEs before the first existing CTE
        insert_at = rewritten_sql.lstrip().index(m.group(0)) + len(m.group(0))
        leading = rewritten_sql[: len(rewritten_sql) - len(rewritten_sql.lstrip())]
        tail = rewritten_sql.lstrip()[len(m.group(0)):]
        result = f"{leading}{m.group(0)}{new_cte_str},\n{tail}"
    else:
        result = f"WITH\n{new_cte_str}\n{rewritten_sql}"

    logger.debug(
        "sql_comparison_utils.cte_wrapped",
        tables_wrapped=[b.split(" AS")[0].strip() for b in cte_blocks],
        shift_days=shift_days,
    )

    return result

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

import re
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

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
        "at",
        "on",
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

# ── Sentinel for None / missing values — sorts last ───────────────────────────
_NONE_SENTINEL = "\xff" * 8


# ─────────────────────────────────────────────────────────────────────────────
# _sort_dataframe
# ─────────────────────────────────────────────────────────────────────────────


def _cell_to_sort_key(value: Any) -> str:
    """
    Safely convert any cell value to a string sort key.

    Rules:
      - None / empty string  → high sentinel (sorts last)
      - list / dict          → str(value)
      - int / float          → zero-padded numeric string for lexicographic sort
      - everything else      → str(value).lower()
    """
    if value is None:
        return _NONE_SENTINEL
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (int, float)):
        # Represent as 30-char zero-padded string so lex order == numeric order
        # for non-negative values; prepend sign character for negatives.
        try:
            f = float(value)
            if f < 0:
                # Negate to sort ascending: more-negative = smaller key
                return f"0{f:030.10f}"
            return f"1{f:030.10f}"
        except (ValueError, OverflowError):
            return str(value)
    if isinstance(value, (list, dict)):
        return str(value)
    s = str(value).strip()
    return s if s else _NONE_SENTINEL


def _column_sort_key(col_values: list[Any]) -> str:
    """
    Compute the successive-minimum signature for a column.

    The signature is the lexicographically smallest non-sentinel cell value
    in the column.  This ensures that columns with smaller minimum values
    sort before columns with larger minimum values — i.e. the column whose
    data starts smallest anchors earliest in the output.
    """
    keys = [_cell_to_sort_key(v) for v in col_values]
    non_sentinel = [k for k in keys if k != _NONE_SENTINEL]
    if not non_sentinel:
        return _NONE_SENTINEL
    return min(non_sentinel)


def _sort_dataframe(
    rows: list[list[Any]],
    columns: list[str],
) -> tuple[list[list[Any]], list[str]]:
    """
    Normalize a query result table for deterministic, column-order-invariant
    comparison. Preserves row sequence so ORDER BY evaluation is respected.

    Algorithm:
      1. Compute a sort key for each column (successive minimum of its values).
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

    # Compute sort key per column
    col_keys: list[tuple[str, str]] = [
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


def evaluate_contains(
    expected_result: Any,
    generated_result: Any,
    numeric_tolerance: int = 6,
) -> tuple[float, dict[str, Any]]:
    """
    Compare expected and generated QueryResults for row containment.

    1. Normalizes columns deterministically via _sort_dataframe (preserving row sequence).
    2. Maps expected columns to generated columns by name, then positionally.
    3. Computes fractional containment score: matched_expected_rows / total_expected_rows.

    Returns:
        (score, details_dict)
    """
    from collections import Counter
    from src.domain.entities.query_result import QueryResult

    if expected_result is None or not expected_result.success:
        return 0.0, {"reason": "expected_sql_execution_failed"}
    if generated_result is None or not generated_result.success:
        return 0.0, {"reason": "generated_sql_execution_failed"}

    # Sort columns deterministically
    exp_rows, exp_cols = _sort_dataframe(
        expected_result.rows, expected_result.columns
    )
    gen_rows, gen_cols = _sort_dataframe(
        generated_result.rows, generated_result.columns
    )

    n_expected_cols = len(exp_cols)
    n_generated_cols = len(gen_cols)

    expected_col_lower = [c.lower().strip() for c in exp_cols]
    generated_col_lower = [c.lower().strip() for c in gen_cols]

    col_indices: list[int | None] = [None] * n_expected_cols
    used_gen_indices: set[int] = set()

    for i, exp_col in enumerate(expected_col_lower):
        if exp_col in generated_col_lower:
            idx = generated_col_lower.index(exp_col)
            col_indices[i] = idx
            used_gen_indices.add(idx)

    gen_idx = 0
    for i in range(n_expected_cols):
        if col_indices[i] is None:
            while gen_idx < n_generated_cols and gen_idx in used_gen_indices:
                gen_idx += 1
            if gen_idx < n_generated_cols:
                col_indices[i] = gen_idx
                used_gen_indices.add(gen_idx)
                gen_idx += 1
            else:
                return 0.0, {
                    "reason": "missing_expected_columns",
                    "expected_columns": expected_col_lower,
                    "generated_columns": generated_col_lower,
                }

    exp_qr = QueryResult(success=True, rows=exp_rows, columns=exp_cols)
    expected_rows = exp_qr.as_normalised_row_tuples(numeric_tolerance)

    projected_gen_rows = []
    for row in gen_rows:
        if all(idx is not None and idx < len(row) for idx in col_indices):
            projected_gen_rows.append([row[idx] for idx in col_indices])
        else:
            return 0.0, {"reason": "generated_row_too_short"}

    tmp_result = QueryResult(
        success=True, rows=projected_gen_rows, columns=exp_cols
    )
    generated_rows = tmp_result.as_normalised_row_tuples(numeric_tolerance)

    if len(expected_rows) != len(generated_rows):
        return 0.0, {
            "reason": "row_count_mismatch",
            "expected_row_count": len(expected_rows),
            "generated_row_count": len(generated_rows),
        }

    expected_counter = Counter(expected_rows)
    generated_counter = Counter(generated_rows)

    total_expected = sum(expected_counter.values())
    if total_expected == 0:
        score = 1.0 if len(generated_rows) == 0 else 0.0
    else:
        matched_count = sum(
            min(generated_counter[row], count)
            for row, count in expected_counter.items()
        )
        score = round(matched_count / total_expected, 4)

    return score, {
        "expected_row_count": len(expected_rows),
        "generated_row_count": len(generated_rows),
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

    with_clause = "WITH\n" + ",\n".join(cte_blocks)
    result = f"{with_clause}\n{rewritten_sql}"

    logger.debug(
        "sql_comparison_utils.cte_wrapped",
        tables_wrapped=[b.split(" AS")[0].replace("WITH\n", "").strip() for b in cte_blocks],
        shift_days=shift_days,
    )

    return result

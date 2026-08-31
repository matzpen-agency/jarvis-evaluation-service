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
        return False


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
      2. Sort columns ascending by that key (ties broken by original position).
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

    # Compute sort key per column — (full_signature, original_position).
    # Using the original index as the tie-breaker instead of column name makes
    # the ordering name-independent: two columns with identical value signatures
    # are interchangeable, so any stable ordering between them is valid.
    col_keys: list[tuple[tuple[_Comparable, ...], int]] = [
        (_column_sort_key(col_values[c]), c)
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


def _is_temporal_type(db_type: str) -> bool:
    """Return True if the DB type indicates a temporal column."""
    db_type = db_type.lower()
    if "date" in db_type or "time" in db_type or "timestamp" in db_type:
        return True
    if "unix_seconds" in db_type or "unix_millis" in db_type or "epoch" in db_type:
        return True
    return False


def _generate_shift_expression(col: str, db_type: str, shift_days: int) -> str:
    """Generate the dialect-specific shift expression for a temporal column."""
    db_type = db_type.lower()
    
    if "unix_millis" in db_type or "epoch_millis" in db_type:
        return f"({col} + ({shift_days} * 86400000))"
    
    if "unix_seconds" in db_type or "unix_timestamp" in db_type or "epoch" in db_type:
        return f"({col} + ({shift_days} * 86400))"
        
    if "iso_timestamp" in db_type or "timestamp_string" in db_type or "varchar" in db_type:
        return f"CAST(date_add('day', {shift_days}, try_cast({col} AS TIMESTAMP)) AS VARCHAR)"
        
    if db_type == "date":
        return f"CAST(date_add('day', {shift_days}, CAST({col} AS DATE)) AS DATE)"
        
    # Default for timestamp, timestampz, datetime, time, etc.
    # Note: We must strip complex type modifiers (like timestamp(3)) if we want to cast back precisely,
    # but Trino supports casting to parameterized types.
    return f"CAST(date_add('day', {shift_days}, CAST({col} AS TIMESTAMP)) AS {db_type})"


def dynamically_wrap_with_yaml_cte(
    sql_query: str,
    shift_days: int,
    table_schema_map: dict[str, dict[str, str]],
) -> str:
    """
    Wrap SQL table references with Trino CTEs that shift date/timestamp columns.

    For each table referenced in sql_query:
      - Looks up its columns in table_schema_map.
      - Identifies temporal columns by exact DB type from the schema map.
      - Generates a CTE that applies cast-shift-cast back to those columns.
      - Safely replaces the table reference in the AST with the CTE alias.

    If shift_days == 0, the original query is returned unchanged.
    If a table has no date columns or is absent from table_schema_map, it is
    left unrewritten.

    Args:
        sql_query:        Original SQL string.
        shift_days:       Number of days to shift (positive = future, negative = past).
        table_schema_map: Mapping of table_name (lower) → dict of col_name → db_type.
                          Obtain from BackendTableResolver.get_table_schema_map().

    Returns:
        Modified SQL with CTEs prepended, or original SQL if no changes needed.
    """
    if shift_days == 0:
        return sql_query

    import sqlglot
    import sqlglot.expressions as exp

    try:
        parsed = sqlglot.parse_one(sql_query, read="trino")
    except Exception as exc:
        logger.debug("sql_comparison_utils.sqlglot_parse_failed", error=str(exc))
        return sql_query

    cte_counter = 1
    tables_wrapped: list[str] = []
    
    # We will accumulate CTE ASTs to add
    ctes_to_add = []

    # Find all table references
    for table_node in list(parsed.find_all(exp.Table)):
        table_name = table_node.name.lower()
        if not table_name:
            continue

        # Look up schema
        schema_dict = table_schema_map.get(table_name, {})
        if not schema_dict:
            # Try to resolve schema/catalog prefixes if they exist in the AST node
            db_name = table_node.db.lower() if table_node.db else ""
            catalog_name = table_node.catalog.lower() if table_node.catalog else ""
            
            candidates = []
            if catalog_name and db_name:
                candidates.append(f"{catalog_name}.{db_name}.{table_name}")
            if db_name:
                candidates.append(f"{db_name}.{table_name}")
            
            for cand in candidates:
                if cand in table_schema_map:
                    schema_dict = table_schema_map[cand]
                    break
                    
        if not schema_dict:
            continue
            
        temporal_cols = {col for col, dtype in schema_dict.items() if _is_temporal_type(dtype)}
        
        if not temporal_cols:
            continue
            
        cte_name = f"__shifted_{table_name}_{cte_counter}"
        cte_counter += 1
        
        # Build the SELECT elements for the CTE using type-aware shift logic
        select_parts = []
        for col, dtype in schema_dict.items():
            if col in temporal_cols:
                shift_expr = _generate_shift_expression(col, dtype, shift_days)
                select_parts.append(f"{shift_expr} AS {col}")
            else:
                select_parts.append(col)
                
        # Build the CTE query string and parse it to AST
        full_table_name = table_node.sql(dialect="trino") 
        cte_query_str = f"SELECT {', '.join(select_parts)} FROM {full_table_name}"
        
        try:
            cte_query_ast = sqlglot.parse_one(cte_query_str, read="trino")
        except Exception as exc:
            logger.error("sql_comparison_utils.cte_parse_failed", table=table_name, error=str(exc))
            continue
            
        ctes_to_add.append((cte_name, cte_query_ast))
        
        tables_wrapped.append(table_name)
        
        # Replace the original table reference with the CTE name, preserving the original alias
        original_alias = table_node.alias
        new_alias = original_alias if original_alias else table_node.name
        
        new_table_node = exp.Table(
            this=exp.to_identifier(cte_name),
            alias=exp.TableAlias(this=exp.to_identifier(new_alias))
        )
        
        table_node.replace(new_table_node)

    if not ctes_to_add:
        return sql_query
        
    # Inject CTEs into the AST using .with_()
    for cte_name, cte_query_ast in ctes_to_add:
        parsed = parsed.with_(cte_name, as_=cte_query_ast, append=True)

    logger.debug(
        "sql_comparison_utils.cte_wrapped",
        tables_wrapped=tables_wrapped,
        shift_days=shift_days,
    )
    
    return parsed.sql(dialect="trino")

"""
sql_comparison_utils.py — Shared utilities for deterministic SQL result comparison.

  normalize_cell(value)
      Canonical form of one result cell so the same value compares equal no
      matter how the engine/driver serialised it (Decimal vs float, "58247" vs
      58247, "$1,234.50", ISO timestamps vs dates, ...).

  compare_results(expected, generated, allow_extra_columns=...)
      Column-order-invariant comparison of two result sets. Columns are matched
      by their data (names are ignored); rows must match as a multiset, or in
      sequence when the expected SQL has an outer ORDER BY (ties tolerated).
      Contains accuracy allows extra generated columns, exact accuracy doesn't.

  dynamically_wrap_with_yaml_cte(sql, shift_days, table_schema_map, dialect)
      Wraps each table reference with a CTE that shifts every date/timestamp
      column by shift_days, rendered for the connection's SQL dialect.
"""

from __future__ import annotations

import datetime as dt
import math
import re
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

import sqlglot
import sqlglot.expressions as exp
import structlog

logger = structlog.get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Parsing helpers
# ─────────────────────────────────────────────────────────────────────────────


def node_arg(node: exp.Expression, name: str):
    """Read an AST arg across sqlglot versions (newer ones renamed e.g. from → from_)."""
    value = node.args.get(name)
    return value if value is not None else node.args.get(f"{name}_")


def safe_parse(sql: str | None, *dialects: str | None) -> exp.Expression | None:
    """Parse sql with the first dialect that succeeds (None = sqlglot default)."""
    if not sql:
        return None
    for dialect in [*dialects, None]:
        try:
            parsed = sqlglot.parse_one(sql, read=dialect)
        except Exception:
            continue
        if parsed is not None:
            return parsed
    return None


def _requires_order_by(sql: str | None, dialect: str | None = None) -> bool:
    """
    Check if the given SQL query contains an outermost ORDER BY clause using sqlglot AST.
    """
    parsed = safe_parse(sql, dialect)
    return parsed is not None and parsed.args.get("order") is not None


def order_key_indices(
    sql: str | None, result_columns: list[str], dialect: str | None = None
) -> list[int] | None:
    """
    Map the outer ORDER BY keys of `sql` to positions in its result columns.

    Used to compare ordered results while tolerating ties: rows that share the
    same sort key may legitimately come back in any order. Returns None when
    the query has no ORDER BY or any key can't be mapped to an output column
    (callers then fall back to strict positional comparison).
    """
    parsed = safe_parse(sql, dialect)
    if parsed is None or parsed.args.get("order") is None:
        return None

    projections = list(parsed.selects) if isinstance(parsed, exp.Query) else []
    has_star = any(p.is_star for p in projections)
    names = [c.lower() for c in result_columns]
    positional = not has_star and len(projections) == len(result_columns)

    indices: list[int] = []
    for ordered in parsed.args["order"].expressions:
        key = ordered.this
        pos: int | None = None

        if isinstance(key, exp.Literal) and key.is_int:
            pos = int(key.this) - 1
        elif isinstance(key, exp.Column) and not key.table and names.count(key.name.lower()) == 1:
            pos = names.index(key.name.lower())

        if pos is None and positional:
            key_sql = key.sql(dialect=dialect).lower()
            for i, projection in enumerate(projections):
                inner = projection.this if isinstance(projection, exp.Alias) else projection
                if inner.sql(dialect=dialect).lower() == key_sql:
                    pos = i
                    break

        if pos is None or not 0 <= pos < len(result_columns):
            return None
        indices.append(pos)
    return indices


def transpile_sql(sql: str, read: str | None, write: str | None) -> str | None:
    """
    Translate SQL between dialects and drop catalog qualifiers
    (e.g. Trino's postgres.public.t → public.t). Returns None if it can't parse.
    """
    try:
        parsed = sqlglot.parse_one(sql, read=read)
    except Exception:
        return None
    if parsed is None:
        return None
    for table in parsed.find_all(exp.Table):
        if table.args.get("catalog"):
            table.set("catalog", None)
    try:
        return parsed.sql(dialect=write)
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Cell normalisation
# ─────────────────────────────────────────────────────────────────────────────

_PLAIN_NUMBER_RE = re.compile(r"[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?")
# "$1,234.50", "-$12.00", "1,234,567", "€ 99.9"
_FORMATTED_NUMBER_RE = re.compile(
    r"([-+]?)\s*[$€£₪]?\s*([-+]?)(\d{1,3}(?:,\d{3})+|\d+)(\.\d+)?"
)
_ISO_DATETIME_RE = re.compile(
    r"(\d{4}-\d{2}-\d{2})"
    r"(?:[T ](\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?))?"
    r"\s*(Z|[+-]\d{2}(?::?\d{2})?)?"
)
_ZERO_TIME_RE = re.compile(r"00:00(?::00(?:\.0+)?)?")
_UTC_SUFFIXES = (None, "Z", "+00", "+00:00", "+0000")


def _normalize_number(value: float, numeric_tolerance: int) -> float | str:
    if math.isnan(value):
        return "nan"
    if math.isinf(value):
        return value
    # Round to the tolerance, then to 12 significant digits so that large
    # values computed slightly differently (float vs numeric) still agree.
    rounded = float(f"{round(value, numeric_tolerance):.12g}")
    return 0.0 if rounded == 0 else rounded


def _parse_number(text: str) -> float | None:
    if _PLAIN_NUMBER_RE.fullmatch(text):
        return float(text)
    match = _FORMATTED_NUMBER_RE.fullmatch(text)
    if match and (any(ch in text for ch in "$€£₪") or "," in text):
        sign = -1.0 if "-" in (match.group(1) + match.group(2)) else 1.0
        return sign * float(match.group(3).replace(",", "") + (match.group(4) or ""))
    return None


def _normalize_iso_datetime(text: str) -> str | None:
    match = _ISO_DATETIME_RE.fullmatch(text)
    if not match:
        return None
    date_part, time_part, tz_part = match.groups()
    if (time_part is None or _ZERO_TIME_RE.fullmatch(time_part)) and tz_part in _UTC_SUFFIXES:
        return date_part
    time_part = re.sub(r"\.0+$", "", time_part or "00:00:00")
    if len(time_part) == 5:
        time_part += ":00"
    suffix = "" if tz_part in _UTC_SUFFIXES else tz_part
    return f"{date_part} {time_part}{suffix}"


def normalize_cell(value: Any, numeric_tolerance: int = 6) -> Any:
    """
    Canonical, hashable form of a result cell.

      - NULL stays None (distinct from the empty string)
      - bool/int/float/Decimal and numeric strings ("58247", "1.50",
        "$27,307,607.39") → float rounded to `numeric_tolerance` decimals
      - dates/timestamps and ISO strings → "YYYY-MM-DD[ HH:MM:SS]"
        (a midnight timestamp equals the plain date)
      - other strings → trimmed, lower-cased
    """
    if value is None:
        return None
    if isinstance(value, bool | int | float | Decimal):
        return _normalize_number(float(value), numeric_tolerance)
    if isinstance(value, dt.datetime):
        return _normalize_iso_datetime(value.isoformat(sep=" ")) or value.isoformat()
    if isinstance(value, dt.date | dt.time):
        return value.isoformat()
    if isinstance(value, bytes):
        return value.hex()
    if isinstance(value, str):
        text = value.strip()
        number = _parse_number(text)
        if number is not None:
            return _normalize_number(number, numeric_tolerance)
        iso = _normalize_iso_datetime(text)
        if iso is not None:
            return iso
        return text.lower()
    return str(value).strip().lower()


# ─────────────────────────────────────────────────────────────────────────────
# Result comparison
# ─────────────────────────────────────────────────────────────────────────────

_MAX_MAPPING_SEARCH_STEPS = 20_000

_SUBSET_RE = re.compile(r"subset(?:\s*,\s*(>=|=)\s*,\s*(\d+))?")


@dataclass(frozen=True)
class Grading:
    """
    How a question's result is graded (BIRD-style grading_method):

      default   rows as a multiset; in order when the golden SQL has ORDER BY
      set       distinct rows, any order (duplicates don't matter)
      multiset  rows with duplicates, any order
      list      rows in the golden order
      subset    generated rows must be a subset of the golden rows, e.g.
                "subset,=,3" — any 3 of the valid answers ("name 3 movies...")
    """

    mode: str = "default"
    op: str | None = None
    count: int | None = None


def parse_grading_method(value: str | None) -> Grading:
    """Parse a grading_method string; unknown/empty values mean default grading."""
    text = (value or "").strip().lower()
    if text in ("set", "multiset", "list"):
        return Grading(mode=text)
    match = _SUBSET_RE.fullmatch(text)
    if match:
        op, count = match.groups()
        return Grading(mode="subset", op=op, count=int(count) if count else None)
    return Grading()


def _rows_in_expected_order(
    expected_rows: list[tuple],
    generated_rows: list[tuple],
    key_indices: list[int] | None,
) -> bool:
    """True when generated rows follow the expected ORDER BY sequence."""
    if generated_rows == expected_rows:
        return True
    if not key_indices:
        return False

    expected_keys = [tuple(r[k] for k in key_indices) for r in expected_rows]
    generated_keys = [tuple(r[k] for k in key_indices) for r in generated_rows]
    if expected_keys != generated_keys:
        return False

    # Same key sequence; rows within a run of equal keys may be in any order
    start = 0
    for i in range(1, len(expected_keys) + 1):
        if i == len(expected_keys) or expected_keys[i] != expected_keys[start]:
            if Counter(expected_rows[start:i]) != Counter(generated_rows[start:i]):
                return False
            start = i
    return True


def _find_column_mapping(
    expected_rows: list[tuple],
    generated_rows: list[tuple],
    requires_ordering: bool,
    key_indices: list[int] | None,
    subset: bool = False,
) -> tuple[list[int] | None, str]:
    """
    Find which generated column holds each expected column's data such that
    the projected generated rows equal the expected rows (or, with subset,
    are all among the expected rows).

    Columns are first paired by identical value multisets (value subsets
    with subset), then a backtracking search picks an assignment whose rows
    line up (needed when several columns hold the same multiset of values,
    e.g. two 0/1 flags).

    Returns (mapping, failure_reason); mapping[i] is the generated column
    index for expected column i.
    """
    n_exp = len(expected_rows[0])
    n_gen = len(generated_rows[0])
    exp_cols = list(zip(*expected_rows, strict=True))
    gen_cols = list(zip(*generated_rows, strict=True))

    candidates: list[list[int]] = []
    if subset:
        gen_values = [set(col) for col in gen_cols]
        for col in exp_cols:
            values = set(col)
            matches = [j for j in range(n_gen) if gen_values[j] <= values]
            if not matches:
                return None, "expected_column_not_found_in_generated"
            candidates.append(matches)
    else:
        gen_counters = [Counter(col) for col in gen_cols]
        for col in exp_cols:
            counter = Counter(col)
            matches = [j for j in range(n_gen) if gen_counters[j] == counter]
            if not matches:
                return None, "expected_column_not_found_in_generated"
            candidates.append(matches)

    search_order = sorted(range(n_exp), key=lambda i: len(candidates[i]))
    assignment: dict[int, int] = {}
    used: set[int] = set()
    steps = 0
    order_failed = False

    def partial_rows_match() -> bool:
        exp_idx = list(assignment.keys())
        gen_idx = [assignment[i] for i in exp_idx]
        if subset:
            exp_set = {tuple(row[i] for i in exp_idx) for row in expected_rows}
            return all(tuple(row[j] for j in gen_idx) in exp_set for row in generated_rows)
        exp_proj = Counter(tuple(row[i] for i in exp_idx) for row in expected_rows)
        gen_proj = Counter(tuple(row[j] for j in gen_idx) for row in generated_rows)
        return exp_proj == gen_proj

    def search(depth: int) -> bool:
        nonlocal steps, order_failed
        if depth == n_exp:
            if not requires_ordering:
                return True
            projected = [
                tuple(row[assignment[i]] for i in range(n_exp)) for row in generated_rows
            ]
            if _rows_in_expected_order(expected_rows, projected, key_indices):
                return True
            order_failed = True
            return False

        exp_i = search_order[depth]
        for gen_j in candidates[exp_i]:
            if gen_j in used:
                continue
            steps += 1
            if steps > _MAX_MAPPING_SEARCH_STEPS:
                return False
            assignment[exp_i] = gen_j
            used.add(gen_j)
            if partial_rows_match() and search(depth + 1):
                return True
            del assignment[exp_i]
            used.discard(gen_j)
        return False

    if search(0):
        return [assignment[i] for i in range(n_exp)], ""
    if order_failed:
        return None, "row_order_mismatch"
    if subset:
        return None, "generated_rows_not_in_expected"
    return None, "projected_rows_do_not_match"


def _distinct(rows: list[tuple]) -> list[tuple]:
    return list(dict.fromkeys(rows))


def _compare_subset(
    expected_rows: list[tuple],
    generated_rows: list[tuple],
    grading: Grading,
    details: dict[str, Any],
) -> tuple[float, dict[str, Any]]:
    """Generated (distinct) rows must all be valid answers, in the required number."""
    expected_rows = _distinct(expected_rows)
    generated_rows = _distinct(generated_rows)
    n = len(generated_rows)
    if n == 0:
        return 0.0, {**details, "reason": "no_generated_rows"}
    if grading.count is not None:
        ok = n == grading.count if grading.op == "=" else n >= grading.count
        if not ok:
            return 0.0, {
                **details,
                "reason": "row_count_mismatch",
                "required_rows": f"{grading.op}{grading.count}",
            }
    if not expected_rows:
        return 0.0, {**details, "reason": "generated_rows_not_in_expected"}
    mapping, reason = _find_column_mapping(
        expected_rows, generated_rows, False, None, subset=True
    )
    if mapping is None:
        return 0.0, {**details, "reason": reason}
    return 1.0, {**details, "column_mapping_indices": mapping}


def compare_results(
    expected_result: Any,
    generated_result: Any,
    *,
    allow_extra_columns: bool,
    requires_ordering: bool = False,
    order_keys: list[int] | None = None,
    numeric_tolerance: int = 6,
    grading: Grading | None = None,
) -> tuple[float, dict[str, Any]]:
    """
    Compare two QueryResults, ignoring column names and column order.

    The generated result matches when its rows equal the expected rows
    (as a multiset, or in sequence when requires_ordering) after projecting
    it onto the columns that hold the expected data. With
    allow_extra_columns the generated result may carry additional columns.
    A per-question grading (see Grading) overrides how rows are compared.

    Returns (1.0 | 0.0, details). details["inconclusive"] is True when the
    comparison can't be decided (both results truncated by the executor).
    """
    grading = grading or Grading()
    if grading.mode in ("set", "multiset", "subset"):
        requires_ordering = False
    elif grading.mode == "list":
        requires_ordering = True
    if expected_result is None or not expected_result.success:
        return 0.0, {
            "reason": "expected_sql_execution_failed",
            "error": getattr(expected_result, "error", None),
        }
    if generated_result is None or not generated_result.success:
        return 0.0, {
            "reason": "generated_sql_execution_failed",
            "error": getattr(generated_result, "error", None),
        }

    n_exp_cols = len(expected_result.columns)
    n_gen_cols = len(generated_result.columns)
    details: dict[str, Any] = {
        "expected_row_count": len(expected_result.rows),
        "generated_row_count": len(generated_result.rows),
        "expected_column_count": n_exp_cols,
        "generated_column_count": n_gen_cols,
        "requires_ordering": requires_ordering,
    }
    if grading.mode != "default":
        details["grading"] = grading.mode

    if n_gen_cols < n_exp_cols:
        return 0.0, {**details, "reason": "insufficient_generated_columns"}
    if not allow_extra_columns and n_gen_cols != n_exp_cols:
        return 0.0, {**details, "reason": "column_count_mismatch"}

    if expected_result.truncated or generated_result.truncated:
        details["expected_truncated"] = expected_result.truncated
        details["generated_truncated"] = generated_result.truncated
        if grading.mode in ("set", "subset"):
            # Distinct-row and subset grading can't be judged from a slice
            return 0.0, {**details, "reason": "results_truncated", "inconclusive": True}
        if expected_result.truncated and generated_result.truncated:
            exp_total = expected_result.total_row_count
            gen_total = generated_result.total_row_count
            if exp_total is not None and gen_total is not None and exp_total != gen_total:
                details["expected_total_rows"] = exp_total
                details["generated_total_rows"] = gen_total
                return 0.0, {**details, "reason": "row_count_mismatch"}
            # Both hit the row cap: the visible rows are an arbitrary slice
            return 0.0, {**details, "reason": "results_truncated", "inconclusive": True}
        # Only one side hit the cap, so its true row count is strictly larger
        return 0.0, {**details, "reason": "row_count_mismatch"}

    exp_rows = expected_result.as_normalised_row_tuples(numeric_tolerance)
    gen_rows = generated_result.as_normalised_row_tuples(numeric_tolerance)

    if grading.mode == "subset":
        score, sub_details = _compare_subset(exp_rows, gen_rows, grading, details)
        indices = sub_details.pop("column_mapping_indices", None)
        if indices is not None:
            sub_details["column_mapping"] = {
                expected_result.columns[i]: generated_result.columns[j]
                for i, j in enumerate(indices)
            }
        return score, sub_details
    if grading.mode == "set":
        exp_rows, gen_rows = _distinct(exp_rows), _distinct(gen_rows)

    if len(exp_rows) != len(gen_rows):
        return 0.0, {**details, "reason": "row_count_mismatch"}

    if not exp_rows:
        return 1.0, {**details, "empty_result": True}

    mapping, reason = _find_column_mapping(
        exp_rows, gen_rows, requires_ordering, order_keys
    )
    if mapping is None:
        return 0.0, {**details, "reason": reason}

    return 1.0, {
        **details,
        "column_mapping": {
            expected_result.columns[i]: generated_result.columns[j]
            for i, j in enumerate(mapping)
        },
    }


def evaluate_contains(
    expected_result: Any,
    generated_result: Any,
    numeric_tolerance: int = 6,
    requires_ordering: bool = False,
    order_keys: list[int] | None = None,
    grading: Grading | None = None,
) -> tuple[float, dict[str, Any]]:
    """
    Contains check: every expected column's data is present in the generated
    result (extra generated columns allowed) and the rows match 100%.
    """
    score, details = compare_results(
        expected_result,
        generated_result,
        allow_extra_columns=True,
        requires_ordering=requires_ordering,
        order_keys=order_keys,
        numeric_tolerance=numeric_tolerance,
        grading=grading,
    )
    if score == 1.0:
        details["column_containment"] = True
    return score, details


# ─────────────────────────────────────────────────────────────────────────────
# dynamically_wrap_with_yaml_cte
# ─────────────────────────────────────────────────────────────────────────────


def _is_temporal_type(db_type: str) -> bool:
    """Return True if the DB type holds a calendar date/instant (not time-of-day)."""
    db_type = db_type.lower().strip()
    if "unix_seconds" in db_type or "unix_millis" in db_type or "epoch" in db_type:
        return True
    if "interval" in db_type:
        return False
    if db_type.startswith("time") and not db_type.startswith("timestamp"):
        return False  # time of day: shifting by whole days is a no-op
    return "date" in db_type or "timestamp" in db_type


def _generate_shift_expression(
    col: str, db_type: str, shift_days: int, dialect: str | None = "trino"
) -> str:
    """Generate the dialect-specific shift expression for a temporal column."""
    db_type = db_type.lower()

    if "unix_millis" in db_type or "epoch_millis" in db_type:
        return f"({col} + ({shift_days} * 86400000))"

    if "unix_seconds" in db_type or "unix_timestamp" in db_type or "epoch" in db_type:
        return f"({col} + ({shift_days} * 86400))"

    if dialect in (None, "trino", "presto"):
        if "iso_timestamp" in db_type or "timestamp_string" in db_type or "varchar" in db_type:
            return f"CAST(date_add('day', {shift_days}, try_cast({col} AS TIMESTAMP)) AS VARCHAR)"
        if db_type == "date":
            return f"CAST(date_add('day', {shift_days}, CAST({col} AS DATE)) AS DATE)"
        # timestamp, timestamp(3), timestamp with time zone, ...
        return f"CAST(date_add('day', {shift_days}, CAST({col} AS TIMESTAMP)) AS {db_type})"

    # Other dialects: <col> + INTERVAL '<n>' DAY, rendered by sqlglot
    shifted: exp.Expression = exp.Add(
        this=sqlglot.parse_one(col, read=dialect),
        expression=exp.Interval(
            this=exp.Literal.string(str(shift_days)), unit=exp.var("DAY")
        ),
    )
    if db_type == "date":
        shifted = exp.Cast(this=exp.paren(shifted), to=exp.DataType.build("date"))
    return shifted.sql(dialect=dialect)


def _lookup_table_schema(
    table_node: exp.Table, table_schema_map: dict[str, dict[str, str]]
) -> dict[str, str]:
    name = table_node.name.lower()
    db_name = (table_node.db or "").lower()
    catalog_name = (table_node.catalog or "").lower()
    candidates = []
    if catalog_name and db_name:
        candidates.append(f"{catalog_name}.{db_name}.{name}")
    if db_name:
        candidates.append(f"{db_name}.{name}")
    candidates.append(name)
    for key in candidates:
        if key in table_schema_map:
            return table_schema_map[key]
    return {}


def dynamically_wrap_with_yaml_cte(
    sql_query: str,
    shift_days: int,
    table_schema_map: dict[str, dict[str, str]],
    dialect: str | None = "trino",
) -> str:
    """
    Wrap SQL table references with CTEs that shift date/timestamp columns.

    For each distinct table referenced in sql_query that has temporal columns
    in table_schema_map, a CTE re-selects every column with the temporal ones
    shifted by shift_days, and the table reference is replaced by the CTE
    (keeping the original alias). The CTEs are prepended so that the query's
    own CTEs can read them.

    Returns the original SQL unchanged when shift_days == 0, when it can't be
    parsed, or when no referenced table has temporal columns.
    """
    if shift_days == 0:
        return sql_query

    def shift(col_sql: str, dtype: str) -> str | None:
        if not _is_temporal_type(dtype):
            return None
        return _generate_shift_expression(col_sql, dtype, shift_days, dialect)

    return _wrap_tables_with_ctes(
        sql_query, table_schema_map, dialect, "__shifted", shift, shift_days=shift_days
    )


def _is_money_type(db_type: str) -> bool:
    return (db_type or "").strip().lower() == "money"


def cast_money_to_numeric(
    sql_query: str,
    table_schema_map: dict[str, dict[str, str]],
    dialect: str | None = "postgres",
) -> str:
    """
    Re-read every referenced table with its money columns cast to NUMERIC.

    Postgres money can't be averaged, compared with numbers or divided like
    one (avg(money), money = integer…), so golden SQL written as if amounts
    were numbers fails. Same CTE technique as the time shift; returns the SQL
    unchanged when no referenced table has a money column.
    """

    def to_numeric(col_sql: str, dtype: str) -> str | None:
        return f"CAST({col_sql} AS NUMERIC)" if _is_money_type(dtype) else None

    return _wrap_tables_with_ctes(sql_query, table_schema_map, dialect, "__numeric", to_numeric)


def _wrap_tables_with_ctes(
    sql_query: str,
    table_schema_map: dict[str, dict[str, str]],
    dialect: str | None,
    cte_prefix: str,
    rewrite_column: Callable[[str, str], str | None],
    **log_fields: Any,
) -> str:
    """
    Replace each referenced table that has a column rewrite_column changes
    by a CTE re-selecting all its columns (rewritten ones replaced), keeping
    the table's alias. rewrite_column(col_sql, db_type) returns the new
    expression, or None to keep the column as is.
    """
    try:
        parsed = sqlglot.parse_one(sql_query, read=dialect)
    except Exception as exc:
        logger.debug("sql_comparison_utils.sqlglot_parse_failed", error=str(exc))
        return sql_query
    if parsed is None:
        return sql_query

    query_cte_names = {cte.alias_or_name.lower() for cte in parsed.find_all(exp.CTE)}
    wrapped_ctes: dict[str, exp.CTE | None] = {}

    for table_node in list(parsed.find_all(exp.Table)):
        table_name = table_node.name.lower()
        if not table_name or (not table_node.db and table_name in query_cte_names):
            continue

        full_table_name = exp.table_(
            table_node.name, db=table_node.db or None, catalog=table_node.catalog or None
        ).sql(dialect=dialect)
        cte_key = full_table_name.lower()

        if cte_key not in wrapped_ctes:
            wrapped_ctes[cte_key] = None
            schema_dict = _lookup_table_schema(table_node, table_schema_map)
            select_parts = []
            rewritten = False
            for col, dtype in schema_dict.items():
                # Postgres folds unquoted names to lower case
                quote = dialect in ("postgres", "redshift") and not col.islower()
                col_sql = exp.to_identifier(col, quoted=quote or None).sql(dialect=dialect)
                new_expr = rewrite_column(col_sql, dtype)
                if new_expr is None:
                    select_parts.append(col_sql)
                else:
                    select_parts.append(f"{new_expr} AS {col_sql}")
                    rewritten = True
            if not rewritten:
                continue
            cte_name = f"{cte_prefix}_{table_name}_{sum(1 for c in wrapped_ctes.values() if c) + 1}"
            cte_query_str = f"SELECT {', '.join(select_parts)} FROM {full_table_name}"
            try:
                cte_query_ast = sqlglot.parse_one(cte_query_str, read=dialect)
            except Exception as exc:
                logger.error(
                    "sql_comparison_utils.cte_parse_failed", table=table_name, error=str(exc)
                )
                continue
            wrapped_ctes[cte_key] = exp.CTE(
                this=cte_query_ast, alias=exp.TableAlias(this=exp.to_identifier(cte_name))
            )

        cte = wrapped_ctes[cte_key]
        if cte is None:
            continue
        alias = table_node.alias or table_node.name
        table_node.replace(
            exp.Table(
                this=exp.to_identifier(cte.alias),
                alias=exp.TableAlias(this=exp.to_identifier(alias)),
            )
        )

    new_ctes = [cte for cte in wrapped_ctes.values() if cte is not None]
    if not new_ctes:
        return sql_query

    existing_with = node_arg(parsed, "with")
    if existing_with is not None:
        existing_with.set("expressions", new_ctes + list(existing_with.expressions))
    else:
        with_key = "with_" if "with_" in type(parsed).arg_types else "with"
        parsed.set(with_key, exp.With(expressions=new_ctes))

    logger.debug(
        "sql_comparison_utils.cte_wrapped",
        tables_wrapped=[k for k, c in wrapped_ctes.items() if c is not None],
        cte_prefix=cte_prefix,
        **log_fields,
    )
    return parsed.sql(dialect=dialect)


# ─────────────────────────────────────────────────────────────────────────────
# Shared execution-based evaluation
# ─────────────────────────────────────────────────────────────────────────────


def grading_for(context: Any) -> Grading:
    """The question's grading, from its dataset item metadata (default if none)."""
    metadata = getattr(context.dataset_item, "metadata", None) or {}
    return parse_grading_method(metadata.get("grading_method"))


async def compare_with_order_policy(
    context: Any,
    expected_result: Any,
    generated_result: Any,
    *,
    allow_extra_columns: bool,
    numeric_tolerance: int,
    order_judge: Any = None,
    expected_sql: str | None = None,
) -> tuple[float, dict[str, Any]]:
    """
    Compare results, deciding automatically whether row order counts.

      - explicit per-question grading (e.g. BIRD "list"/"set") is followed as is
      - no outer ORDER BY in the golden SQL → order ignored
      - outer ORDER BY (with or without LIMIT) → rows compared first; if they
        match but come back in another order, the order judge decides from
        the question whether order was asked for (decision + reason in
        details). With LIMIT a wrong sort usually already shows up as wrong
        rows; the judge catches the right top k listed in the wrong order.
        When no decision is possible the result passes, flagged "order not
        verified".
    """
    expected_sql = expected_sql or context.expected_sql
    grading = grading_for(context)
    if grading.mode != "default":
        requires_ordering = grading.mode == "list"
        keys = (
            order_key_indices(expected_sql, expected_result.columns, context.dialect)
            if requires_ordering and expected_result is not None
            else None
        )
        return compare_results(
            expected_result,
            generated_result,
            allow_extra_columns=allow_extra_columns,
            requires_ordering=requires_ordering,
            order_keys=keys,
            numeric_tolerance=numeric_tolerance,
            grading=grading,
        )

    score, details = compare_results(
        expected_result,
        generated_result,
        allow_extra_columns=allow_extra_columns,
        requires_ordering=False,
        numeric_tolerance=numeric_tolerance,
    )
    if not _requires_order_by(expected_sql, context.dialect):
        return score, details
    if score < 1.0 or details.get("inconclusive") or details.get("empty_result"):
        return score, details

    order_keys = order_key_indices(expected_sql, expected_result.columns, context.dialect)
    ordered_score, _ = compare_results(
        expected_result,
        generated_result,
        allow_extra_columns=allow_extra_columns,
        requires_ordering=True,
        order_keys=order_keys,
        numeric_tolerance=numeric_tolerance,
    )
    if ordered_score >= 1.0:
        details["order_policy"] = "order matches"
        return score, details

    decision = await order_judge.requires_order(context.query) if order_judge else None
    if decision is not None and decision.requires_order is not None:
        details["order_judgment"] = {
            "requires_order": decision.requires_order,
            "reason": decision.reason,
            "source": decision.source,
        }
    if decision is not None and decision.requires_order:
        details["order_policy"] = "order differs; the question asks for an order"
        details["reason"] = "row_order_mismatch"
        return 0.0, details
    if decision is not None and decision.requires_order is False:
        details["order_policy"] = "order differs; the question does not ask for an order"
        return score, details
    details["order_policy"] = "order differs; not verified (order judge unavailable)"
    details["order_not_verified"] = True
    if decision is not None:
        details["order_judge_error"] = decision.reason
    return score, details


def has_no_data(result: Any) -> bool:
    """No rows, or only NULLs (e.g. SUM over no rows): nothing to compare."""
    if result is None or getattr(result, "truncated", False):
        return False
    rows = result.rows or []
    return all(cell is None for row in rows for cell in row)


async def evaluate_execution_match(
    context: Any,
    evaluator_name: str,
    *,
    allow_extra_columns: bool,
    numeric_tolerance: int,
    order_judge: Any = None,
) -> Any:
    """
    Score context.generated_result against context.expected_result.

    Not applicable (excluded from averages) when the golden SQL is missing or
    fails to execute, when its result has no data (an empty answer matches
    anything that also finds nothing), or when both results were truncated
    by the executor — those say nothing about the agent.
    """
    from src.domain.entities.evaluation_result import EvaluationResult

    if not context.expected_sql:
        return EvaluationResult.not_applicable(evaluator_name, "no_expected_sql")
    if context.expected_result is None or not context.expected_result.success:
        return EvaluationResult.not_applicable(
            evaluator_name,
            "expected_sql_execution_failed",
            error=getattr(context.expected_result, "error", None),
        )
    if has_no_data(context.expected_result):
        return EvaluationResult.not_applicable(evaluator_name, "golden_result_has_no_data")
    if not context.generated_sql and context.generated_result is None:
        return EvaluationResult(
            evaluator_name=evaluator_name,
            score=0.0,
            passed=False,
            details={"reason": "no_generated_sql", "agent_error": context.error},
        )

    score, details = await compare_with_order_policy(
        context,
        context.expected_result,
        context.generated_result,
        allow_extra_columns=allow_extra_columns,
        numeric_tolerance=numeric_tolerance,
        order_judge=order_judge,
    )
    if details.get("inconclusive"):
        reason = details.pop("reason")
        return EvaluationResult.not_applicable(evaluator_name, reason, **details)
    if allow_extra_columns and score == 1.0:
        details["column_containment"] = True
    return EvaluationResult(
        evaluator_name=evaluator_name,
        score=score,
        passed=score >= 1.0,
        details=details,
    )

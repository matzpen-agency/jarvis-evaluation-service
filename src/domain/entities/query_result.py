"""
query_result.py — Domain entity for SQL query execution results.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class QueryResult:
    """Normalised output from executing a SQL query via any query executor."""

    success: bool
    rows: list[list[Any]] = field(default_factory=list)
    columns: list[str] = field(default_factory=list)
    row_count: int = 0
    execution_time_ms: float = 0.0
    error: str | None = None
    # True when the executor capped the rows (more rows exist than returned).
    truncated: bool = False
    # Full row count of a truncated result, when it was counted separately
    total_row_count: int | None = None

    @classmethod
    def failure(cls, error: str, execution_time_ms: float = 0.0) -> QueryResult:
        """Factory for failed execution results."""
        return cls(
            success=False,
            error=error,
            execution_time_ms=execution_time_ms,
        )

    def as_row_tuples(self) -> list[tuple]:
        """Convert rows to list of tuples for set-based comparison."""
        return [tuple(row) for row in self.rows]

    def as_normalised_row_tuples(self, numeric_tolerance: int = 6) -> list[tuple]:
        """
        Normalise rows for comparison (see normalize_cell): numbers (including
        numeric strings, Decimals and formatted money) become rounded floats,
        timestamps become canonical ISO strings, other strings are trimmed and
        lower-cased, and NULL stays None (distinct from the empty string).
        """
        from src.domain.evaluators.sql_comparison_utils import normalize_cell

        return [
            tuple(normalize_cell(cell, numeric_tolerance) for cell in row)
            for row in self.rows
        ]

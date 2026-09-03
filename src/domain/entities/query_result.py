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
        Normalise rows for comparison:
          - Convert numeric strings to rounded floats
          - Strip leading/trailing whitespace from strings
          - Convert None to empty string
        """
        normalised: list[tuple] = []
        for row in self.rows:
            norm_row: list[Any] = []
            for cell in row:
                if cell is None:
                    norm_row.append("")
                elif isinstance(cell, str):
                    stripped = cell.strip()
                    try:
                        norm_row.append(round(float(stripped), numeric_tolerance))
                    except ValueError:
                        norm_row.append(stripped.lower())
                elif isinstance(cell, float | int):
                    norm_row.append(round(float(cell), numeric_tolerance))
                else:
                    norm_row.append(str(cell).strip().lower())
            normalised.append(tuple(norm_row))
        return normalised

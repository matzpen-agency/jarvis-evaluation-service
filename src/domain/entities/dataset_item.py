"""
dataset_item.py — Generic domain entity for a single evaluation dataset item.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class DatasetItem:
    """
    A single item in an evaluation dataset.

    Fields are intentionally generic to support any dataset schema.
    Concrete use-case code (e.g. TextToSqlDatasetItemParser) extracts
    domain-specific fields from `input` and `expected_output`.
    """

    id: str
    input: dict[str, Any]
    expected_output: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)

    # Convenience accessors for the Text-to-SQL use case
    @property
    def query(self) -> str:
        """Natural language query from the input dict."""
        return self.input.get("query", "")

    @property
    def expected_sql(self) -> str:
        """Expected SQL from the expected_output dict."""
        return self.expected_output.get("sql", "")

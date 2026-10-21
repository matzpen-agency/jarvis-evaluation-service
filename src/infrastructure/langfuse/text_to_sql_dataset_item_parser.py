"""
text_to_sql_dataset_item_parser.py — Parses Langfuse dataset items for Text-to-SQL.

Expected dataset schema:
  {
    "input": { "query": "natural language question" },
    "expected_output": { "sql": "expected SQL query" }
  }

This parser implements DatasetItemParser for this specific schema.
Future schemas (RAG, classification) would have their own parsers.
"""

from __future__ import annotations

from typing import Any

from src.domain.entities.dataset_item import DatasetItem
from src.ports.dataset_item_parser import DatasetItemParser


class DatasetItemParseError(Exception):
    """Raised when a dataset item cannot be parsed."""


class TextToSqlDatasetItemParser(DatasetItemParser):
    """
    Parses dataset items following the Text-to-SQL schema:
      input.query        → DatasetItem.input["query"]
      expected_output.sql → DatasetItem.expected_output["sql"]
    """

    def parse(self, raw: dict[str, Any]) -> DatasetItem:
        item_id = raw.get("id", "")
        inp = raw.get("input", {}) or {}
        exp = raw.get("expected_output", {}) or {}
        meta = raw.get("metadata", {}) or {}

        if not inp.get("query"):
            raise DatasetItemParseError(
                f"Dataset item '{item_id}' missing required field: input.query"
            )

        # Normalise expected SQL — accept "sql" or "response" keys
        expected_sql = exp.get("sql") or exp.get("response") or ""

        return DatasetItem(
            id=item_id,
            input={"query": inp["query"]},
            expected_output={"sql": expected_sql},
            metadata=meta,
            raw=raw,
        )

"""
bird_dataset_loader.py — BIRD-Verified benchmark questions, bundled locally.

Source: wenyupapa/BIRD-Verified-CoT-2462-GPT5.4 (valid-only split): 2462
BIRD train questions over 69 SQLite databases, each with gold SQL that was
verified to return the reference result, an `evidence` hint and a
`grading_method` (set / list / multiset / subset,=,N).

The questions ship with the service (src/config/bird/, see ATTRIBUTION.md)
so evaluations need no internet access.

A question can only be evaluated on a data connection holding its database,
so items are grouped by db_id; datasets are named "bird-<db_id>". The gold
SQL is SQLite — the runner transpiles it to the connection's dialect.
"""

from __future__ import annotations

import json
import re
from functools import cached_property

import structlog

from src.domain.entities.dataset_item import DatasetItem

logger = structlog.get_logger(__name__)

BIRD_DATASET_PREFIX = "bird-"
BIRD_SQL_DIALECT = "sqlite"


def bird_db_id_for_connection(connection_name: str | None) -> str | None:
    """
    Derive the BIRD db_id from a connection name: "movie_platform",
    "bird_movie_platform", "BIRD-movie_platform" → "movie_platform".
    """
    if not connection_name:
        return None
    name = connection_name.strip().lower()
    name = re.sub(r"^bird[\s_\-:]+", "", name)
    return re.sub(r"[\s\-]+", "_", name) or None


class BirdDatasetLoader:
    """Reads the bundled BIRD-Verified JSONL and builds DatasetItems."""

    def __init__(self, path: str, include_evidence: bool = True) -> None:
        self._path = path
        self._include_evidence = include_evidence

    def db_ids(self) -> list[str]:
        return sorted({row["db_id"] for row in self._rows})

    def load(self, db_id: str) -> list[DatasetItem]:
        """All questions for one BIRD database (empty if the db_id is unknown)."""
        return [self._to_item(row) for row in self._rows if row["db_id"] == db_id]

    # ── Internals ─────────────────────────────────────────────────────────────

    @cached_property
    def _rows(self) -> list[dict]:
        try:
            with open(self._path, encoding="utf-8") as f:
                rows = [json.loads(line) for line in f if line.strip()]
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"BIRD questions file not found at {self._path} (BIRD_DATASET_PATH)"
            ) from exc
        logger.info("bird_dataset_loader.loaded", path=self._path, questions=len(rows))
        return rows

    def _to_item(self, row: dict) -> DatasetItem:
        question = (row.get("question") or "").strip()
        evidence = (row.get("evidence") or "").strip()
        query = question
        if self._include_evidence and evidence:
            # BIRD's standard setting: the external-knowledge hint goes with the question
            query = f"{question}\n\nHint: {evidence}"

        return DatasetItem(
            id=f"bird-{row['question_id']}",
            input={"query": query},
            expected_output={"sql": (row.get("SQL") or "").strip()},
            metadata={
                "source": "bird-verified-cot-2462",
                "db_id": row["db_id"],
                "bird_question_id": row["question_id"],
                "grading_method": (row.get("grading_method") or "").strip(),
                "evidence": evidence,
                "sql_dialect": BIRD_SQL_DIALECT,
            },
        )

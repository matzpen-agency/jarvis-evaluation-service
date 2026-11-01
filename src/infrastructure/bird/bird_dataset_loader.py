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

A few databases are bundled too (src/config/bird/databases/<db_id>.sqlite),
loaded into a connection's database by load_bird_databases.py as schemas
"bird_<db_id>". Their questions carry that schema ("sql_schema") and are
limited to those verify_bird_questions.py found to keep their answer on
Trino and Postgres. "bird-all" is every bundled database's questions.

With a dataverse schema (BIRD_DATAVERSE_SCHEMA), every BIRD database was
pushed to dataverse by push_bird_to_oasis.py instead: all of them are
available, "bird-all" holds every question, and the gold SQL uses the
dataverse names, "<schema>".<db_id>__<table> with cleaned column names.
Questions verify_bird_questions.py checked there are limited to those that
keep their answer.
"""

from __future__ import annotations

import json
import re
from functools import cached_property
from pathlib import Path

import structlog

from src.domain.entities.dataset_item import DatasetItem
from src.domain.evaluators.sql_comparison_utils import clean_name

logger = structlog.get_logger(__name__)

BIRD_DATASET_PREFIX = "bird-"
BIRD_ALL_BUNDLED = "all"
BIRD_SQL_DIALECT = "sqlite"
VERIFIED_QUESTIONS_FILE = "verified_questions.json"
# Key in VERIFIED_QUESTIONS_FILE for questions checked on dataverse
DATAVERSE_ENGINE = "dataverse"


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

    def __init__(
        self,
        path: str,
        include_evidence: bool = True,
        databases_dir: str | None = None,
        schema_prefix: str = "bird_",
        dataverse_schema: str = "",
    ) -> None:
        self._path = path
        self._include_evidence = include_evidence
        self._databases_dir = Path(databases_dir) if databases_dir else None
        self._schema_prefix = schema_prefix
        self._dataverse_schema = dataverse_schema

    def db_ids(self) -> list[str]:
        return sorted({row["db_id"] for row in self._rows})

    def bundled_db_ids(self) -> list[str]:
        """
        Databases loaded where evaluations can query them: every one when they
        were pushed to dataverse, else those shipped with the service, ready
        for load_bird_databases.py.
        """
        if self._dataverse_schema:
            return self.db_ids()
        if self._databases_dir is None:
            return []
        return sorted(p.stem for p in self._databases_dir.glob("*.sqlite"))

    def load(self, db_id: str) -> list[DatasetItem]:
        """
        Questions for one BIRD database (empty if the db_id is unknown); for a
        bundled one, only its verified questions, in its schema.
        """
        rows = [row for row in self._rows if row["db_id"] == db_id]
        if db_id not in self.bundled_db_ids():
            return [self._to_item(row) for row in rows]
        verified = self._verified.get(db_id)
        if verified is not None:
            rows = [row for row in rows if row["question_id"] in verified]
        if self._dataverse_schema:
            naming = {
                "sql_schema": self._dataverse_schema,
                "sql_table_prefix": f"{clean_name(db_id)}__",
                "sql_clean_names": True,
            }
        else:
            naming = {"sql_schema": f"{self._schema_prefix}{db_id}".lower()}
        return [self._to_item(row, naming) for row in rows]

    def load_bundled(self) -> list[DatasetItem]:
        return [item for db_id in self.bundled_db_ids() for item in self.load(db_id)]

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

    @cached_property
    def _verified(self) -> dict[str, set[int]]:
        """
        Per database, the questions verified on every engine checked — only
        on dataverse when the databases are there.
        """
        if self._databases_dir is None:
            return {}
        path = self._databases_dir / VERIFIED_QUESTIONS_FILE
        if not path.exists():
            return {}
        per_engine: dict[str, dict[str, list[int]]] = json.loads(path.read_text())
        if self._dataverse_schema:
            per_engine = {DATAVERSE_ENGINE: per_engine.get(DATAVERSE_ENGINE, {})}
        else:
            per_engine.pop(DATAVERSE_ENGINE, None)
        verified: dict[str, set[int]] = {}
        for by_db in per_engine.values():
            for db_id, ids in by_db.items():
                verified[db_id] = verified.get(db_id, set(ids)) & set(ids)
        return verified

    def _to_item(self, row: dict, naming: dict | None = None) -> DatasetItem:
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
                **(naming or {}),
            },
        )

"""
spider2_snow_downloader.py — Downloads and translates Spider 2-snow benchmark questions.

Fetches Snowflake-based questions (sf_ prefix) from the public Spider 2-snow GitHub
repository, translates gold SQL from Snowflake to Trino dialect using sqlglot,
and caches results locally to avoid re-downloading on every run.
"""

from __future__ import annotations

import json
import os
import time
import urllib.request
from typing import Optional

import sqlglot
import structlog

from src.domain.entities.dataset_item import DatasetItem

logger = structlog.get_logger(__name__)

SPIDER2_JSONL_URL = (
    "https://raw.githubusercontent.com/xlang-ai/Spider2/main/"
    "spider2-snow/spider2-snow.jsonl"
)
SPIDER2_GOLD_SQL_BASE_URL = (
    "https://raw.githubusercontent.com/xlang-ai/Spider2/main/"
    "spider2-snow/evaluation_suite/gold/sql/"
)
GITHUB_TIMEOUT = 15  # seconds per request


class Spider2SnowDownloader:
    """
    Downloads Spider 2-snow benchmark questions from GitHub, filters to Snowflake
    databases available in Trino, translates SQL to Trino dialect, and caches results.

    Catalog availability and database filtering are applied dynamically when
    loading from the cache, ensuring the cache file remains complete and reusable.
    """

    def __init__(
        self,
        cache_path: str,
        available_catalogs: Optional[set[str]] = None,
        cache_ttl_hours: int = 24,
        github_enabled: bool = True,
        db_filter: Optional[str] = None,
    ) -> None:
        self._cache_path = cache_path
        self._available_catalogs = available_catalogs  # None = accept all
        self._cache_ttl_seconds = cache_ttl_hours * 3600
        self._github_enabled = github_enabled
        self._db_filter = db_filter.lower() if db_filter else None

    # ── Public API ────────────────────────────────────────────────────────────

    def download(self) -> list[DatasetItem]:
        """
        Return benchmark items. Uses cache if fresh; otherwise re-downloads.
        Falls back to stale cache if GitHub is unreachable.
        """
        if self._github_enabled and self._cache_needs_refresh():
            try:
                # Download and cache ALL compatible items
                all_items = self._fetch_from_github()
                self._write_cache(all_items)
                logger.info(
                    "spider2_snow_downloader.refreshed",
                    count=len(all_items),
                    cache=self._cache_path,
                )
            except Exception as exc:
                logger.warning(
                    "spider2_snow_downloader.github_failed_using_cache",
                    error=str(exc),
                )

        # Always read from the cache and apply filters dynamically
        return self._load_cache()

    # ── GitHub fetching ───────────────────────────────────────────────────────

    def _fetch_from_github(self) -> list[DatasetItem]:
        """Fetch questions + gold SQL from GitHub and return DatasetItems."""
        questions = self._fetch_jsonl()
        sf_questions = [q for q in questions if q.get("instance_id", "").startswith("sf_")]
        logger.info(
            "spider2_snow_downloader.sf_questions_found",
            total=len(questions),
            sf_count=len(sf_questions),
        )

        items: list[DatasetItem] = []
        for q in sf_questions:
            instance_id = q["instance_id"]
            db = q.get("db_id", "")
            catalog = self._db_to_catalog(db)

            gold_sql = self._fetch_gold_sql(instance_id)
            if not gold_sql:
                continue

            trino_sql = self._translate_sql(gold_sql, instance_id)

            item = DatasetItem(
                id=instance_id,
                input={"query": q["instruction"]},
                expected_output={"sql": trino_sql},
                metadata={
                    "difficulty": "complex",
                    "question_type": "join",
                    "source": "spider2_snow",
                    "db": db,
                    "catalog": catalog,
                    "external_knowledge": q.get("external_knowledge"),
                },
            )
            items.append(item)

        logger.info("spider2_snow_downloader.fetch_complete", count=len(items))
        return items

    def _fetch_jsonl(self) -> list[dict]:
        """Download and parse the Spider 2-snow questions JSONL file."""
        req = urllib.request.Request(
            SPIDER2_JSONL_URL,
            headers={"User-Agent": "jarvis-evaluation-service/1.0"},
        )
        with urllib.request.urlopen(req, timeout=GITHUB_TIMEOUT) as resp:
            content = resp.read().decode("utf-8")

        questions = []
        for line in content.splitlines():
            line = line.strip()
            if line:
                try:
                    questions.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        return questions

    def _fetch_gold_sql(self, instance_id: str) -> Optional[str]:
        """Download the gold SQL file for a given instance ID."""
        url = f"{SPIDER2_GOLD_SQL_BASE_URL}{instance_id}.sql"
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "jarvis-evaluation-service/1.0"},
            )
            with urllib.request.urlopen(req, timeout=GITHUB_TIMEOUT) as resp:
                return resp.read().decode("utf-8").strip()
        except Exception as exc:
            logger.debug(
                "spider2_snow_downloader.gold_sql_fetch_failed", url=url, error=str(exc)
            )
            return None

    # ── SQL translation ───────────────────────────────────────────────────────

    @staticmethod
    def _translate_sql(snowflake_sql: str, instance_id: str) -> str:
        """
        Translate Snowflake SQL to Trino SQL using sqlglot.

        Identifiers like "PATENTS"."PATENTS"."APPLICATIONS" become
        patents.patents.applications in Trino dialect.
        """
        try:
            results = sqlglot.transpile(snowflake_sql, read="snowflake", write="trino")
            if results:
                trino_sql = results[0]
                # Clean up sqlglot comma + CROSS JOIN UNNEST bug
                trino_sql = trino_sql.replace(",  CROSS JOIN UNNEST", " CROSS JOIN UNNEST")
                trino_sql = trino_sql.replace(", CROSS JOIN UNNEST", " CROSS JOIN UNNEST")
                trino_sql = trino_sql.replace(",  cross join unnest", " cross join unnest")
                trino_sql = trino_sql.replace(", cross join unnest", " cross join unnest")
                return trino_sql
        except Exception as exc:
            logger.warning(
                "spider2_snow_downloader.sql_translation_failed",
                instance_id=instance_id,
                error=str(exc),
            )
        # Return original SQL as fallback — Trino may still accept it
        return snowflake_sql

    # ── Catalog mapping ───────────────────────────────────────────────────────

    @staticmethod
    def _db_to_catalog(db: str) -> str:
        """
        Map a Spider 2 db name to a Trino catalog name.

        Spider 2 uses uppercase db names like "PATENTS", "GITHUB_REPOS".
        Trino catalog filenames are lowercase (e.g. patents, github_repos).
        The mapping is simply db.lower().
        """
        return db.lower().replace(" ", "_")

    # ── Cache handling ────────────────────────────────────────────────────────

    def _cache_needs_refresh(self) -> bool:
        """Return True if cache is missing or older than TTL."""
        if not os.path.exists(self._cache_path):
            return True
        # If cache is basically empty (e.g. 2 bytes), force refresh
        if os.path.exists(self._cache_path) and os.path.getsize(self._cache_path) < 10:
            return True
        age_seconds = time.time() - os.path.getmtime(self._cache_path)
        return age_seconds > self._cache_ttl_seconds

    def _write_cache(self, items: list[DatasetItem]) -> None:
        """Serialise DatasetItems to the cache JSON file."""
        os.makedirs(os.path.dirname(self._cache_path), exist_ok=True)
        raw = [
            {
                "id": item.id,
                "input": item.input,
                "expected_output": item.expected_output,
                "metadata": item.metadata,
            }
            for item in items
        ]
        with open(self._cache_path, "w", encoding="utf-8") as f:
            json.dump(raw, f, indent=2, ensure_ascii=False)

    def _load_cache(self) -> list[DatasetItem]:
        """Load DatasetItems from the local cache file."""
        if not os.path.exists(self._cache_path):
            logger.error("spider2_snow_downloader.cache_not_found", path=self._cache_path)
            raise FileNotFoundError(
                f"Spider 2-snow cache not found at {self._cache_path} and GitHub is disabled."
            )

        with open(self._cache_path, "r", encoding="utf-8") as f:
            raw_items = json.load(f)

        items = []
        for raw in raw_items:
            catalog = raw.get("metadata", {}).get("catalog", "")

            # Apply db_filter dynamically
            if self._db_filter and catalog != self._db_filter:
                continue

            # Apply catalog availability filter dynamically
            if self._available_catalogs is not None and catalog not in self._available_catalogs:
                continue

            # Prefix the ID if db_filter is active to prevent Langfuse conflicts
            item_id = raw["id"]
            if self._db_filter:
                item_id = f"spider2-{self._db_filter}-{item_id}"

            items.append(
                DatasetItem(
                    id=item_id,
                    input=raw["input"],
                    expected_output=raw["expected_output"],
                    metadata=raw.get("metadata", {}),
                )
            )

        logger.info("spider2_snow_downloader.loaded_from_cache", count=len(items))
        return items
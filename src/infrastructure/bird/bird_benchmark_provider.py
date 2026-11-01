"""
bird_benchmark_provider.py — BIRD-Verified as a BenchmarkProvider.

Requested as dataset "bird" on a connection named after a BIRD database
(e.g. "movie_platform" or "bird_movie_platform"), which resolves to the
dataset "bird-<db_id>" holding that database's questions. On any other
connection whose name starts with "bird" (e.g. "bird"), it resolves to
"bird-all": the questions of every bundled database, loaded into that
connection's database by load_bird_databases.py.
"""

from __future__ import annotations

import asyncio

import structlog

from src.domain.entities.dataset_item import DatasetItem
from src.infrastructure.bird.bird_dataset_loader import (
    BIRD_ALL_BUNDLED,
    BIRD_DATASET_PREFIX,
    BirdDatasetLoader,
    bird_db_id_for_connection,
)
from src.ports.benchmark_provider import BenchmarkProvider
from src.ports.dataset_provider import DatasetNotFoundError

logger = structlog.get_logger(__name__)


class BirdBenchmarkProvider(BenchmarkProvider):
    def __init__(self, loader: BirdDatasetLoader) -> None:
        self._loader = loader

    @property
    def name(self) -> str:
        return "bird"

    def owns(self, dataset_name: str) -> bool:
        return dataset_name.lower().startswith(BIRD_DATASET_PREFIX)

    def dataset_for_connection(self, connection_name: str | None) -> str:
        db_id = bird_db_id_for_connection(connection_name)
        if db_id and db_id in self._loader.db_ids():
            return f"{BIRD_DATASET_PREFIX}{db_id}"
        bundled = self._loader.bundled_db_ids()
        if bundled and (connection_name or "").strip().lower().startswith("bird"):
            return f"{BIRD_DATASET_PREFIX}{BIRD_ALL_BUNDLED}"
        raise DatasetNotFoundError(
            f"Connection {connection_name!r} doesn't hold a BIRD database. Run BIRD on a "
            f"connection named after a BIRD database id, or on a connection whose name "
            f"starts with 'bird' that reads the schemas loaded by load_bird_databases "
            f"(bundled: {', '.join(bundled) or 'none'})."
        )

    async def load(self, dataset_name: str) -> list[DatasetItem]:
        db_id = self._db_id(dataset_name)
        if db_id == BIRD_ALL_BUNDLED:
            items = await asyncio.to_thread(self._loader.load_bundled)
        else:
            items = await asyncio.to_thread(self._loader.load, db_id)
        if not items:
            available = await asyncio.to_thread(self._loader.db_ids)
            raise DatasetNotFoundError(
                f"No BIRD questions for database '{db_id}'. The connection's name must "
                f"be a BIRD database id, one of: {', '.join(available)}"
            )
        logger.info("bird_benchmark.loaded", db_id=db_id, items=len(items))
        return items

    def description(self, dataset_name: str) -> str:
        db_id = self._db_id(dataset_name)
        if db_id == BIRD_ALL_BUNDLED:
            return (
                "BIRD-Verified benchmark questions for the bundled databases: "
                + ", ".join(self._loader.bundled_db_ids())
            )
        return f"BIRD-Verified benchmark questions for database '{db_id}'"

    @staticmethod
    def _db_id(dataset_name: str) -> str:
        return dataset_name[len(BIRD_DATASET_PREFIX) :].strip().lower()

"""
langfuse_dataset_provider.py — Langfuse implementation of DatasetProvider.

Loads dataset items from Langfuse and maps them to the generic DatasetItem
domain entity using the configurable DatasetItemParser.

Benchmark datasets ("bird-<db_id>") are built from their public source
instead and upserted into Langfuse, so evaluation runs can be linked to them.
"""

from __future__ import annotations

import asyncio

import langfuse as lf_sdk
import structlog

from src.domain.entities.dataset_item import DatasetItem
from src.infrastructure.bird.bird_dataset_loader import (
    BIRD_DATASET_PREFIX,
    BirdDatasetLoader,
)
from src.ports.dataset_item_parser import DatasetItemParser
from src.ports.dataset_provider import DatasetProvider

logger = structlog.get_logger(__name__)


class DatasetNotFoundError(Exception):
    """Raised when the requested dataset does not exist in Langfuse."""


class LangfuseDatasetProvider(DatasetProvider):
    """
    Retrieves dataset items from Langfuse and maps them via a parser.

    Gracefully handles Langfuse connectivity issues by raising
    DatasetNotFoundError so the API layer can return a clear 404.
    """

    def __init__(
        self,
        client: lf_sdk.Langfuse,
        parser: DatasetItemParser,
        bird_loader: BirdDatasetLoader | None = None,
    ) -> None:
        self._client = client
        self._parser = parser
        self._bird_loader = bird_loader

    async def get_dataset(self, name: str) -> list[DatasetItem]:
        """
        Load all items from a named Langfuse dataset.

        Raises:
            DatasetNotFoundError: If the dataset does not exist.
        """
        logger.info("langfuse_dataset_provider.loading", dataset_name=name)
        if name.lower().startswith(BIRD_DATASET_PREFIX):
            items = await self._load_bird(name)
            await asyncio.to_thread(
                self._sync_to_langfuse,
                name,
                items,
                f"BIRD-Verified benchmark questions for database "
                f"'{name[len(BIRD_DATASET_PREFIX):]}'",
            )
            return items

        dataset = None
        for attempt in range(3):
            try:
                # Sync SDK call (paginated HTTP) — keep it off the event loop
                dataset = await asyncio.to_thread(self._client.get_dataset, name)
                break
            except Exception as exc:
                error_str = str(exc).lower()
                if "not found" in error_str or "404" in error_str:
                    raise DatasetNotFoundError(
                        f"Dataset '{name}' not found in Langfuse."
                    ) from exc
                if attempt == 2:
                    raise
                logger.warning(
                    "langfuse_dataset_provider.get_dataset_retry",
                    attempt=attempt + 1,
                    error=str(exc),
                )
                await asyncio.sleep(2.0 * (attempt + 1))


        items: list[DatasetItem] = []
        for raw_item in dataset.items:
            try:
                raw_dict = {
                    "id": raw_item.id,
                    "input": raw_item.input or {},
                    "expected_output": raw_item.expected_output or {},
                    "metadata": raw_item.metadata or {},
                }
                parsed = self._parser.parse(raw_dict)
                items.append(parsed)
            except Exception as exc:
                logger.warning(
                    "langfuse_dataset_provider.item_parse_failed",
                    item_id=getattr(raw_item, "id", "unknown"),
                    error=str(exc),
                )

        logger.info(
            "langfuse_dataset_provider.loaded",
            dataset_name=name,
            total_items=len(items),
        )
        return items

    async def _load_bird(self, dataset_name: str) -> list[DatasetItem]:
        if self._bird_loader is None:
            raise DatasetNotFoundError("BIRD datasets are not configured on this service.")
        db_id = dataset_name[len(BIRD_DATASET_PREFIX):].strip().lower()
        items = await asyncio.to_thread(self._bird_loader.load, db_id)
        if not items:
            available = await asyncio.to_thread(self._bird_loader.db_ids)
            raise DatasetNotFoundError(
                f"No BIRD questions for database '{db_id}'. The connection's name must "
                f"be a BIRD database id, one of: {', '.join(available)}"
            )
        logger.info("langfuse_dataset_provider.bird_loaded", db_id=db_id, items=len(items))
        return items

    def _sync_to_langfuse(
        self, dataset_name: str, items: list[DatasetItem], description: str
    ) -> None:
        """
        Upsert locally built dataset items into Langfuse under their own IDs so
        that the run's dataset-run-item links resolve (otherwise linking fails
        with 404 and the experiment shows no items).
        """
        if self._client is None:
            return
        try:
            self._client.create_dataset(name=dataset_name, description=description)
        except Exception:
            pass  # already exists

        synced = 0
        for item in items:
            try:
                self._client.create_dataset_item(
                    dataset_name=dataset_name,
                    id=item.id,
                    input=item.input,
                    expected_output=item.expected_output,
                    metadata=item.metadata,
                )
                synced += 1
            except Exception as exc:
                logger.warning(
                    "langfuse_dataset_provider.sync_item_failed",
                    item_id=item.id,
                    error=str(exc),
                )
        logger.info(
            "langfuse_dataset_provider.synced_to_langfuse",
            dataset_name=dataset_name,
            synced=synced,
            total=len(items),
        )

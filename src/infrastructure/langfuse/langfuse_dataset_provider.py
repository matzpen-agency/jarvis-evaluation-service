"""
langfuse_dataset_provider.py — Langfuse implementation of DatasetProvider.

Loads dataset items from Langfuse and maps them to the generic DatasetItem
domain entity using the configurable DatasetItemParser.

Benchmark datasets (e.g. "bird-<db_id>") come from the configured
BenchmarkProvider instead and are upserted into Langfuse, so evaluation runs
can be linked to them.
"""

from __future__ import annotations

import asyncio

import langfuse as lf_sdk
import structlog

from src.domain.entities.dataset_item import DatasetItem
from src.ports.benchmark_provider import BenchmarkProvider
from src.ports.dataset_item_parser import DatasetItemParser
from src.ports.dataset_provider import DatasetNotFoundError, DatasetProvider

logger = structlog.get_logger(__name__)

__all__ = ["DatasetNotFoundError", "LangfuseDatasetProvider"]


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
        benchmark: BenchmarkProvider | None = None,
    ) -> None:
        self._client = client
        self._parser = parser
        self._benchmark = benchmark

    async def get_dataset(self, name: str) -> list[DatasetItem]:
        """
        Load all items from a named Langfuse dataset.

        Raises:
            DatasetNotFoundError: If the dataset does not exist.
        """
        logger.info("langfuse_dataset_provider.loading", dataset_name=name)
        if self._benchmark is not None and self._benchmark.owns(name):
            items = await self._benchmark.load(name)
            await asyncio.to_thread(
                self._sync_to_langfuse, name, items, self._benchmark.description(name)
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

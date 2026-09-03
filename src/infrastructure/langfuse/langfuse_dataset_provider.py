"""
langfuse_dataset_provider.py — Langfuse implementation of DatasetProvider.

Loads dataset items from Langfuse and maps them to the generic DatasetItem
domain entity using the configurable DatasetItemParser.
"""

from __future__ import annotations

import langfuse as lf_sdk
import structlog

from src.domain.entities.dataset_item import DatasetItem
from src.ports.dataset_item_parser import DatasetItemParser
from src.ports.dataset_provider import DatasetProvider

from src.ports.query_executor import QueryExecutor

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
        query_executor: QueryExecutor | None = None,
    ) -> None:
        self._client = client
        self._parser = parser
        self._query_executor = query_executor

    async def get_dataset(self, name: str) -> list[DatasetItem]:
        """
        Load all items from a named Langfuse dataset.

        Raises:
            DatasetNotFoundError: If the dataset does not exist.
        """
        logger.info("langfuse_dataset_provider.loading", dataset_name=name)
        if "spider2" in name.lower() or name == "spider":
            logger.info("langfuse_dataset_provider.spider2_force_local", dataset_name=name)
            items = await self._load_spider2_benchmark(name)
            self._sync_to_langfuse(name, items)
            return items

        try:
            dataset = self._client.get_dataset(name)
        except Exception as exc:
            error_str = str(exc).lower()
            if "not found" in error_str or "404" in error_str:
                raise DatasetNotFoundError(
                    f"Dataset '{name}' not found in Langfuse."
                ) from exc
            raise


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

    async def _load_spider2_benchmark(self, dataset_name: str) -> list[DatasetItem]:
        import asyncio
        import os
        from src.config.settings import Settings
        from src.infrastructure.spider2.spider2_snow_downloader import Spider2SnowDownloader

        settings = Settings()
        path = settings.SPIDER2_QUESTIONS_PATH
        if not os.path.isabs(path):
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
            path = os.path.abspath(os.path.join(project_root, path))

        # Query available catalogs from Trino
        available_catalogs = None
        if self._query_executor:
            try:
                res = await self._query_executor.execute("SHOW CATALOGS")
                if res.success and res.rows:
                    available_catalogs = {row[0].lower() for row in res.rows if row}
                    logger.info("langfuse_dataset_provider.available_catalogs", catalogs=available_catalogs)
            except Exception as exc:
                logger.warning("langfuse_dataset_provider.fetch_catalogs_failed", error=str(exc))

        # Check if the dataset name requests a specific database (e.g. "spider2-airlines")
        db_filter = None
        if "-" in dataset_name:
            parts = dataset_name.split("-", 1)
            if len(parts) > 1 and parts[1].strip():
                db_filter = parts[1].strip()

        downloader = Spider2SnowDownloader(
            cache_path=path,
            available_catalogs=available_catalogs,
            cache_ttl_hours=settings.SPIDER2_CACHE_TTL_HOURS,
            github_enabled=settings.SPIDER2_GITHUB_ENABLED,
            db_filter=db_filter,
        )

        items = await asyncio.to_thread(downloader.download)
        return items

    def _sync_to_langfuse(self, dataset_name: str, items: list[DatasetItem]) -> None:
        """
        Upsert local dataset items into Langfuse so that _link_to_dataset_run
        can find them by ID. Without this, linking always fails with 404 and
        the Langfuse Datasets view shows no SQL and status=error.

        Uses the same local item ID (e.g. 'spider2-airlines-1') so that the
        CreateDatasetRunItemRequest.datasetItemId matches.
        """
        if self._client is None:
            return
        try:
            # Create the dataset if it doesn't already exist
            try:
                self._client.create_dataset(
                    name=dataset_name,
                    description=f"Spider2.0 evaluation benchmark ({dataset_name})",
                )
                logger.info("langfuse_dataset_provider.dataset_created", dataset_name=dataset_name)
            except Exception:
                # Dataset already exists — that's fine
                pass

            # Upsert every item using the local ID so linking works
            synced = 0
            for item in items:
                try:
                    self._client.create_dataset_item(
                        dataset_name=dataset_name,
                        id=item.id,
                        input={"query": item.query},
                        expected_output={"sql": item.expected_sql},
                        metadata={"source": "local_spider2_json"},
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
        except Exception as exc:
            logger.warning(
                "langfuse_dataset_provider.sync_to_langfuse_failed",
                dataset_name=dataset_name,
                error=str(exc),
            )

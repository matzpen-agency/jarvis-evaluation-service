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
    ) -> None:
        self._client = client
        self._parser = parser

    async def get_dataset(self, name: str) -> list[DatasetItem]:
        """
        Load all items from a named Langfuse dataset.

        Raises:
            DatasetNotFoundError: If the dataset does not exist.
        """
        logger.info("langfuse_dataset_provider.loading", dataset_name=name)
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

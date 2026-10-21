"""
dataset_item_parser.py — Abstract interface for parsing raw dataset items.

Allows different dataset schemas to be supported without changing orchestration.

Implementations: TextToSqlDatasetItemParser
Future:         RagDatasetItemParser, ClassificationDatasetItemParser, ...
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from src.domain.entities.dataset_item import DatasetItem


class DatasetItemParser(ABC):
    """
    Converts raw dataset item dicts (from any source) into typed DatasetItem
    domain objects.

    Implement this to support new dataset schemas without touching the runner.
    """

    @abstractmethod
    def parse(self, raw: dict[str, Any]) -> DatasetItem:
        """
        Parse a raw dataset item dict into a DatasetItem.

        Args:
            raw: Raw dict from the dataset provider (e.g. Langfuse item as dict).

        Returns:
            Typed DatasetItem with validated fields.

        Raises:
            DatasetItemParseError: If required fields are missing or malformed.
        """
        ...

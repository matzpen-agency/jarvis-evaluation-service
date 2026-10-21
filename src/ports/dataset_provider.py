"""
dataset_provider.py — Abstract interface for loading evaluation datasets.

Implementations: LangfuseDatasetProvider
Future:         PromptfooDatasetProvider, FileDatasetProvider, ...
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from src.domain.entities.dataset_item import DatasetItem


class DatasetProvider(ABC):
    """
    Generic interface for retrieving evaluation datasets.

    A dataset is a named collection of (input, expected_output) pairs.
    Implementations handle the specifics of the storage backend.
    """

    @abstractmethod
    async def get_dataset(self, name: str) -> list[DatasetItem]:
        """
        Load all items from a named dataset.

        Args:
            name: Dataset identifier (e.g., Langfuse dataset name).

        Returns:
            Ordered list of DatasetItems ready for evaluation.

        Raises:
            DatasetNotFoundError: If the dataset does not exist.
            DatasetProviderError: On connection or parsing failures.
        """
        ...

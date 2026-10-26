"""
benchmark_provider.py — Abstract interface for a public text-to-SQL benchmark.

A benchmark ships its own questions and gold SQL with the service (no internet
access needed), grouped per database. A user asks for it by its short name
(e.g. "bird") on a data connection; the benchmark maps that connection to one
of its databases and supplies that database's questions as a dataset.

Implementations: BirdBenchmarkProvider
To use another benchmark: implement this port and return it from
_get_benchmark_provider() in src/api/dependencies/container.py.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from src.domain.entities.dataset_item import DatasetItem


class BenchmarkProvider(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        """The dataset name users request to run this benchmark (e.g. "bird")."""
        ...

    @abstractmethod
    def owns(self, dataset_name: str) -> bool:
        """Whether a resolved dataset name (e.g. "bird-movie_platform") is ours."""
        ...

    @abstractmethod
    def dataset_for_connection(self, connection_name: str | None) -> str:
        """
        The dataset name for the database a connection holds, found by the
        connection's name.

        Raises:
            DatasetNotFoundError: If the name doesn't identify one of our databases.
        """
        ...

    @abstractmethod
    async def load(self, dataset_name: str) -> list[DatasetItem]:
        """
        The questions of one of our datasets. Items may carry metadata the
        runner and evaluators understand: "sql_dialect" (the gold SQL's
        dialect, transpiled to the connection's) and "grading_method".

        Raises:
            DatasetNotFoundError: If there is no such dataset.
        """
        ...

    @abstractmethod
    def description(self, dataset_name: str) -> str:
        """Human-readable description, used for the dataset in Langfuse."""
        ...

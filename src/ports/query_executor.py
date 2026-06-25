"""
query_executor.py — Abstract interface for executing SQL queries.

Implementations: TrinoQueryExecutor
Future:         SnowflakeQueryExecutor, BigQueryQueryExecutor, DuckDBQueryExecutor, ...
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from src.domain.entities.query_result import QueryResult


class QueryExecutor(ABC):
    """
    Generic interface for executing SQL and capturing structured results.

    Implementations are responsible for:
      - Connection management
      - Async wrapping of synchronous drivers
      - Error normalisation into QueryResult
    """

    @abstractmethod
    async def execute(self, sql: str) -> QueryResult:
        """
        Execute a SQL query and return structured results.

        Args:
            sql: SQL statement to execute (SELECT only for safety).

        Returns:
            QueryResult with rows, columns, timing and error info.
            Never raises on SQL errors — errors are captured in QueryResult.
        """
        ...

"""
query_executor.py — Abstract interface for executing SQL queries.

Implementations: TrinoQueryExecutor, JeenQueryExecutor
AgentSqlExecutor implementations: AgentTrinoQueryExecutor
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
    async def execute(self, sql: str, connection_id: int | None = None) -> QueryResult:
        """
        Execute a SQL query and return structured results.

        Args:
            sql: SQL statement to execute (SELECT only for safety).
            connection_id: Data connection to run against. Executors bound to a
                single database ignore it.

        Returns:
            QueryResult with rows, columns, timing and error info.
            Never raises on SQL errors — errors are captured in QueryResult.
        """
        ...


class AgentSqlExecutor(QueryExecutor):
    """
    Runs the agent's generated SQL on the engine the agent itself queried
    (e.g. its Trino catalog), for the connections that engine covers.

    Column types can differ between that engine and the connection's own
    database (Postgres money is varchar in Trino), so SQL the agent wrote and
    tested there may not run on the database directly.
    """

    @abstractmethod
    def covers(self, connection_id: int | None) -> bool:
        """Whether the agent's engine serves this connection's data."""
        ...

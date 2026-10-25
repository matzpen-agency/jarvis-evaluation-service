"""
agent_trino_query_executor.py — Runs the agent's SQL on the agent's Trino.

Each covered connection maps to the Trino catalog that holds its data
(AGENT_TRINO_CATALOGS, e.g. "6=adventureworks"); other connections are not
covered and keep running on the connection itself.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from src.domain.entities.query_result import QueryResult
from src.infrastructure.trino.trino_query_executor import TrinoQueryExecutor
from src.ports.query_executor import AgentSqlExecutor


def parse_catalog_map(value: str) -> dict[int, str]:
    """"6=adventureworks, 7=sales" → {6: "adventureworks", 7: "sales"}."""
    catalogs: dict[int, str] = {}
    for part in value.split(","):
        if not part.strip():
            continue
        connection_id, sep, catalog = part.partition("=")
        if not sep or not catalog.strip():
            raise ValueError(f"expected <connection_id>=<catalog>, got {part.strip()!r}")
        catalogs[int(connection_id.strip())] = catalog.strip()
    return catalogs


class AgentTrinoQueryExecutor(AgentSqlExecutor):
    def __init__(
        self,
        connection_factory: Callable[[str], Any],
        catalogs: dict[int, str],
        max_rows: int | None = None,
    ) -> None:
        self._executors = {
            connection_id: TrinoQueryExecutor(
                connection_factory=lambda catalog=catalog: connection_factory(catalog),
                max_rows=max_rows,
            )
            for connection_id, catalog in catalogs.items()
        }

    def covers(self, connection_id: int | None) -> bool:
        return connection_id in self._executors

    async def execute(self, sql: str, connection_id: int | None = None) -> QueryResult:
        executor = self._executors.get(connection_id)  # type: ignore[arg-type]
        if executor is None:
            return QueryResult.failure(f"agent Trino has no catalog for connection {connection_id}")
        return await executor.execute(sql)

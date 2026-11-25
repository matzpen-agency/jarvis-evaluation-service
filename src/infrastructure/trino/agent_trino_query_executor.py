"""
agent_trino_query_executor.py — Runs the agent's SQL on the agent's Trino.

Like the agent, every connection runs in one catalog (AGENT_TRINO_CATALOG,
the agent's TRINO_CATALOG); AGENT_TRINO_CATALOGS (e.g. "6=adventureworks")
overrides it per connection. Without a default catalog only the listed
connections are covered; the others keep running on the connection itself.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

from src.domain.entities.query_result import QueryResult
from src.infrastructure.trino.trino_query_executor import TrinoQueryExecutor
from src.ports.query_executor import AgentSqlExecutor


def parse_catalog_map(value: str) -> dict[int, str]:
    """ "6=adventureworks, 7=sales" → {6: "adventureworks", 7: "sales"}."""
    catalogs: dict[int, str] = {}
    for part in value.split(","):
        if not part.strip():
            continue
        connection_id, sep, catalog = part.partition("=")
        if not sep or not catalog.strip():
            raise ValueError(
                f"expected <connection_id>=<catalog>, got {part.strip()!r}"
            )
        catalogs[int(connection_id.strip())] = catalog.strip()
    return catalogs


class AgentTrinoQueryExecutor(AgentSqlExecutor):
    def __init__(
        self,
        connection_factory: Callable[[str], Any],
        catalogs: dict[int, str],
        max_rows: int | None = None,
        default_catalog: str | None = None,
        max_concurrent_queries: int = 2,
        busy_retry_delay_seconds: float = 3.0,
    ) -> None:
        # The databases behind the agent's Trino have few connections: the
        # time shift alone sends 10 queries per question, which used up their
        # connection slots and left most shifts inconclusive
        self._semaphore = asyncio.Semaphore(max(1, max_concurrent_queries))
        self._busy_retry_delay = busy_retry_delay_seconds
        self._catalogs = catalogs
        self._default_catalog = default_catalog or None
        self._executors = {
            catalog: TrinoQueryExecutor(
                connection_factory=lambda catalog=catalog: connection_factory(catalog),
                max_rows=max_rows,
            )
            for catalog in {*catalogs.values(), *filter(None, [self._default_catalog])}
        }

    def catalog_for(self, connection_id: int | None) -> str | None:
        return self._catalogs.get(connection_id, self._default_catalog)  # type: ignore[arg-type]

    def covers(self, connection_id: int | None) -> bool:
        return self.catalog_for(connection_id) is not None

    async def execute(self, sql: str, connection_id: int | None = None) -> QueryResult:
        catalog = self.catalog_for(connection_id)
        if catalog is None:
            return QueryResult.failure(
                f"agent Trino has no catalog for connection {connection_id}"
            )
        async with self._semaphore:
            result = await self._executors[catalog].execute(sql)
            if not result.success and _out_of_connections(result.error):
                await asyncio.sleep(self._busy_retry_delay)
                result = await self._executors[catalog].execute(sql)
            return result


def _out_of_connections(error: str | None) -> bool:
    """The database behind Trino had no free connection (try again shortly)."""
    text = (error or "").lower()
    return "connection slots are reserved" in text or "too many connections" in text

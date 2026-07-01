"""
trino_query_executor.py — Trino implementation of QueryExecutor.

Executes SQL queries against Trino using the synchronous DBAPI driver,
wrapped in asyncio.to_thread() to avoid blocking the event loop.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from typing import Any

import structlog

from src.domain.entities.query_result import QueryResult
from src.ports.query_executor import QueryExecutor

logger = structlog.get_logger(__name__)


class TrinoQueryExecutor(QueryExecutor):
    """
    Executes SQL against Trino.

    A new connection is created per query to keep the implementation
    stateless and safe under concurrent use. Connection pooling can be
    added later without changing the interface.

    When TRINO_ENABLED=False, queries are short-circuited with an empty
    successful result, matching the behaviour in text2sql-onboarding.
    """

    def __init__(
        self,
        connection_factory: Callable[[], Any],
        enabled: bool = True,
    ) -> None:
        self._connection_factory = connection_factory
        self._enabled = enabled

    async def execute(self, sql: str) -> QueryResult:
        """
        Execute a SQL query asynchronously.

        Delegates to a thread to avoid blocking the event loop.
        """
        if not self._enabled:
            logger.warning("trino_query_executor.disabled", sql_preview=sql[:80])
            return QueryResult(
                success=True,
                rows=[],
                columns=[],
                row_count=0,
                execution_time_ms=0.0,
                error="Trino disabled (TRINO_ENABLED=False)",
            )

        return await asyncio.to_thread(self._execute_sync, sql)

    def _execute_sync(self, sql: str) -> QueryResult:
        """Synchronous Trino DBAPI execution — runs in a thread pool."""
        start = time.monotonic()
        conn = None
        cur = None
        # Trino's DBAPI rejects trailing semicolons with SYNTAX_ERROR
        sql = sql.strip().rstrip(";").strip()
        log_sql = sql.replace("\n", " ")[:300]
        logger.debug("trino_query_executor.executing", sql=log_sql)

        try:
            conn = self._connection_factory()
            cur = conn.cursor()
            cur.execute(sql)
            rows = cur.fetchall()
            columns = [desc[0] for desc in cur.description] if cur.description else []
            elapsed_ms = (time.monotonic() - start) * 1000

            logger.debug(
                "trino_query_executor.success",
                rows=len(rows),
                elapsed_ms=round(elapsed_ms, 1),
            )
            return QueryResult(
                success=True,
                rows=[list(r) for r in rows],
                columns=columns,
                row_count=len(rows),
                execution_time_ms=round(elapsed_ms, 2),
            )

        except Exception as exc:
            elapsed_ms = (time.monotonic() - start) * 1000
            error_msg = str(exc)
            logger.error(
                "trino_query_executor.failed",
                error=error_msg,
                elapsed_ms=round(elapsed_ms, 1),
            )
            return QueryResult.failure(error=error_msg, execution_time_ms=elapsed_ms)

        finally:
            for resource in (cur, conn):
                if resource is not None:
                    try:
                        resource.close()
                    except Exception:
                        pass

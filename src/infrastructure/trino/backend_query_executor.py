"""
backend_query_executor.py — REST API implementation of QueryExecutor.

Queries Trino via the text2sql-onboarding backend API.
"""

from __future__ import annotations

import httpx
import structlog

from src.domain.entities.query_result import QueryResult
from src.ports.query_executor import QueryExecutor

logger = structlog.get_logger(__name__)


class BackendQueryExecutor(QueryExecutor):
    """
    Executes SQL queries by delegating to the backend REST API.
    """

    def __init__(
        self,
        backend_url: str,
        query_endpoint: str = "/api/query/execute",
        timeout: float = 30.0,
        enabled: bool = True,
    ) -> None:
        self._url = f"{backend_url.rstrip('/')}/{query_endpoint.lstrip('/')}"
        self._timeout = timeout
        self._enabled = enabled

    async def execute(self, sql: str) -> QueryResult:
        """
        Execute SQL query via the backend REST API.
        """
        if not self._enabled:
            logger.warning("backend_query_executor.disabled", sql_preview=sql[:80])
            return QueryResult(
                success=True,
                rows=[],
                columns=[],
                row_count=0,
                execution_time_ms=0.0,
                error="Query execution disabled (TRINO_ENABLED=False)",
            )

        payload = {"sql": sql}
        logger.debug("backend_query_executor.executing", url=self._url, sql_preview=sql[:80])

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(self._url, json=payload)
                response.raise_for_status()
                data = response.json()

            return QueryResult(
                success=data.get("success", False),
                rows=data.get("rows", []),
                columns=data.get("columns", []),
                row_count=data.get("row_count", 0),
                execution_time_ms=data.get("execution_time_ms", 0.0),
                error=data.get("error"),
            )

        except httpx.TimeoutException:
            logger.error("backend_query_executor.timeout", url=self._url)
            return QueryResult.failure(
                error=f"Query execution timed out after {self._timeout}s",
                execution_time_ms=self._timeout * 1000,
            )

        except httpx.HTTPStatusError as exc:
            error_msg = f"Backend returned HTTP {exc.response.status_code}: {exc.response.text[:200]}"
            logger.error("backend_query_executor.http_error", status=exc.response.status_code)
            return QueryResult.failure(error=error_msg)

        except Exception as exc:
            logger.error("backend_query_executor.unexpected_error", error=str(exc))
            return QueryResult.failure(error=f"Query execution failed: {exc}")

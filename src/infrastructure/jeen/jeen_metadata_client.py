"""
jeen_metadata_client.py — Jeen metadata MCP client for data connections.

Golden questions are curated per Jeen connection, and a connection can be any
engine (Postgres, Trino, Databricks, ...). Jeen is the only component that can
reach every connection, so expected and generated SQL are executed through its
`run_query` tool, and schema/dialect come from `get_columns` / `list_connections`.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import httpx
import structlog

from src.domain.entities.query_result import QueryResult
from src.infrastructure.mcp.mcp_http_client import McpHttpClient, McpProtocolError
from src.ports.connection_metadata_provider import ConnectionMetadataProvider
from src.ports.query_executor import QueryExecutor

logger = structlog.get_logger(__name__)

# Jeen connection dialect → sqlglot dialect
_SQLGLOT_DIALECTS: dict[str, str] = {
    "postgresql": "postgres",
    "postgres": "postgres",
    "trino": "trino",
    "presto": "presto",
    "spark": "databricks",
    "databricks": "databricks",
    "mysql": "mysql",
    "mariadb": "mysql",
    "tsql": "tsql",
    "mssql": "tsql",
    "snowflake": "snowflake",
    "bigquery": "bigquery",
    "oracle": "oracle",
    "duckdb": "duckdb",
    "sqlite": "sqlite",
    "redshift": "redshift",
}

_TRANSIENT_ERRORS = (httpx.TransportError, McpProtocolError)
# Jeen rate-limits run_query; keep trying (with longer waits) before giving up
_RATE_LIMIT_ATTEMPTS = 6


class JeenMetadataClient:
    """Thin, cached wrapper over the Jeen metadata MCP tools."""

    def __init__(
        self,
        url: str,
        api_key: str,
        timeout: float = 120.0,
        max_rows: int = 1000,
        max_concurrency: int = 4,
        max_retries: int = 3,
        cache_ttl_seconds: float = 600.0,
    ) -> None:
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._mcp = McpHttpClient(url, headers=headers, timeout=timeout)
        self._max_rows = max_rows
        self._max_retries = max_retries
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._cache_ttl = cache_ttl_seconds
        self._cache: dict[str, tuple[float, Any]] = {}
        self._cache_lock = asyncio.Lock()

    # ── Public API ────────────────────────────────────────────────────────────

    async def run_query(self, connection_id: int, sql: str) -> QueryResult:
        """Execute SQL on the connection. SQL errors are returned, never raised."""
        sql = sql.strip().rstrip(";").strip()
        start = time.monotonic()
        try:
            result = await self._call(
                "run_query",
                {"connection_id": connection_id, "query": sql, "max_rows": self._max_rows},
            )
        except Exception as exc:
            elapsed_ms = (time.monotonic() - start) * 1000
            logger.error("jeen_metadata.run_query_failed", error=str(exc))
            return QueryResult.failure(f"Jeen run_query failed: {exc}", elapsed_ms)

        elapsed_ms = round((time.monotonic() - start) * 1000, 2)
        if result.is_error:
            return QueryResult.failure(result.text or "query failed", elapsed_ms)

        try:
            payload = result.json()
        except ValueError:
            return QueryResult.failure(
                f"Unexpected run_query response: {(result.text or '')[:200]}",
                elapsed_ms,
            )

        columns = [str(c) for c in payload.get("columns") or []]
        if len(set(columns)) != len(columns):
            # run_query returns rows as objects, so duplicate names collapse
            logger.warning("jeen_metadata.duplicate_column_names", columns=columns)
        rows = [
            [row.get(c) for c in columns] if isinstance(row, dict) else list(row)
            for row in payload.get("rows") or []
        ]
        return QueryResult(
            success=True,
            rows=rows,
            columns=columns,
            row_count=len(rows),
            execution_time_ms=elapsed_ms,
            truncated=bool(payload.get("truncated")),
        )

    async def get_connection(self, connection_id: int) -> dict[str, Any] | None:
        connections = await self._cached("connections", self._list_connections)
        for conn in connections:
            if conn.get("connection_id") == connection_id:
                return conn
        return None

    async def get_dialect(self, connection_id: int) -> str | None:
        conn = await self.get_connection(connection_id)
        if not conn:
            return None
        return _SQLGLOT_DIALECTS.get(str(conn.get("dialect") or "").lower())

    async def get_table_schema_map(self, connection_id: int) -> dict[str, dict[str, str]]:
        return await self._cached(
            f"schema:{connection_id}", lambda: self._load_schema_map(connection_id)
        )

    # ── Internals ─────────────────────────────────────────────────────────────

    async def _list_connections(self) -> list[dict[str, Any]]:
        result = await self._call("list_connections", {})
        if result.is_error:
            raise RuntimeError(f"list_connections failed: {result.text}")
        payload = result.json()
        return payload.get("connections", []) if isinstance(payload, dict) else payload

    async def _load_schema_map(self, connection_id: int) -> dict[str, dict[str, str]]:
        result = await self._call("get_columns", {"connection_id": connection_id})
        if result.is_error:
            raise RuntimeError(f"get_columns failed: {result.text}")
        payload = result.json()
        columns = payload.get("columns", []) if isinstance(payload, dict) else payload

        schema_map: dict[str, dict[str, str]] = {}
        for col in columns:
            table = str(col.get("table") or "").lower().strip()
            # Keep the column's case: SQL generated from it (time-shift CTEs)
            # must quote mixed-case Postgres columns exactly
            name = str(col.get("column_name") or "").strip()
            if not table or not name:
                continue
            db_type = str(col.get("data_type") or "varchar").lower()
            schema = str(col.get("schema") or "").lower().strip()
            catalog = str(col.get("catalog") or "").lower().strip()

            keys = [table]
            if schema:
                keys.append(f"{schema}.{table}")
                if catalog:
                    keys.append(f"{catalog}.{schema}.{table}")
            for key in keys:
                schema_map.setdefault(key, {})[name] = db_type
        return schema_map

    async def _cached(self, key: str, loader) -> Any:
        async with self._cache_lock:
            hit = self._cache.get(key)
            if hit and time.monotonic() - hit[0] < self._cache_ttl:
                return hit[1]
            value = await loader()
            self._cache[key] = (time.monotonic(), value)
            return value

    async def _call(self, tool: str, arguments: dict[str, Any]):
        """
        Call a tool, retrying transport failures and 5xx with short backoff,
        and rate limiting (429) with longer waits honoring Retry-After.
        """
        last_exc: Exception | None = None
        rate_limited = False
        attempt = 0
        while True:
            attempt += 1
            try:
                async with self._semaphore:
                    return await self._mcp.call_tool(tool, arguments)
            except httpx.HTTPStatusError as exc:
                last_exc = exc
                status = exc.response.status_code
                if status < 500 and status != 429:
                    raise
                rate_limited = status == 429
            except _TRANSIENT_ERRORS as exc:
                last_exc = exc
                rate_limited = False

            max_attempts = _RATE_LIMIT_ATTEMPTS if rate_limited else self._max_retries
            if attempt >= max_attempts:
                break
            if rate_limited:
                wait = _retry_after_seconds(last_exc) or min(30.0, 5.0 * attempt)
            else:
                wait = 1.5 * attempt
            logger.warning(
                "jeen_metadata.retry",
                tool=tool,
                attempt=attempt,
                rate_limited=rate_limited,
                error=str(last_exc),
                next_retry_in_s=wait,
            )
            await asyncio.sleep(wait)
        assert last_exc is not None
        raise last_exc


def _retry_after_seconds(exc: Exception | None) -> float | None:
    """Seconds from a 429's Retry-After header (capped at 60), if present."""
    response = getattr(exc, "response", None)
    value = response.headers.get("retry-after") if response is not None else None
    try:
        return min(60.0, max(0.0, float(value))) if value else None
    except ValueError:
        return None


class JeenConnectionMetadataProvider(ConnectionMetadataProvider):
    """Connection metadata from Jeen; empty/None when a lookup fails."""

    def __init__(self, client: JeenMetadataClient) -> None:
        self._client = client

    async def get_dialect(self, connection_id: int | None) -> str | None:
        if connection_id is None:
            return None
        try:
            return await self._client.get_dialect(connection_id)
        except Exception as exc:
            logger.warning("jeen_metadata.dialect_lookup_failed", error=str(exc))
            return None

    async def get_connection_name(self, connection_id: int | None) -> str | None:
        if connection_id is None:
            return None
        try:
            conn = await self._client.get_connection(connection_id)
        except Exception as exc:
            logger.warning("jeen_metadata.connection_lookup_failed", error=str(exc))
            return None
        return (conn or {}).get("name")

    async def get_table_schema_map(
        self, connection_id: int | None = None
    ) -> dict[str, dict[str, str]]:
        if connection_id is None:
            return {}
        try:
            return await self._client.get_table_schema_map(connection_id)
        except Exception as exc:
            logger.warning("jeen_metadata.schema_lookup_failed", error=str(exc))
            return {}


class JeenQueryExecutor(QueryExecutor):
    """
    Runs SQL on the item's Jeen connection; falls back to another executor
    (e.g. direct Trino) when no connection_id is given.
    """

    def __init__(
        self, client: JeenMetadataClient, fallback: QueryExecutor | None = None
    ) -> None:
        self._client = client
        self._fallback = fallback

    async def execute(self, sql: str, connection_id: int | None = None) -> QueryResult:
        if connection_id is not None:
            return await self._client.run_query(connection_id, sql)
        if self._fallback is not None:
            return await self._fallback.execute(sql)
        return QueryResult.failure("No connection_id given and no fallback executor")

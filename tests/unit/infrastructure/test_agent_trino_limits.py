"""
test_agent_trino_limits.py — The agent's Trino gets few queries at once, and a
query the database turned away for lack of connections is retried.
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

from src.domain.entities.query_result import QueryResult
from src.infrastructure.trino.agent_trino_query_executor import AgentTrinoQueryExecutor

BUSY = (
    'TrinoExternalError(type=EXTERNAL, name=JDBC_ERROR, message="FATAL: remaining '
    'connection slots are reserved for roles with the SUPERUSER attribute")'
)


def _executor(execute, **kwargs) -> AgentTrinoQueryExecutor:
    executor = AgentTrinoQueryExecutor(
        connection_factory=MagicMock(), catalogs={6: "adventureworks"}, **kwargs
    )
    executor._executors["adventureworks"].execute = execute
    return executor


@pytest.mark.unit
@pytest.mark.asyncio
async def test_at_most_the_limit_run_at_once():
    running = peak = 0

    async def execute(sql):
        nonlocal running, peak
        running += 1
        peak = max(peak, running)
        await asyncio.sleep(0.01)
        running -= 1
        return QueryResult(success=True)

    executor = _executor(execute, max_concurrent_queries=2)
    await asyncio.gather(
        *(executor.execute("SELECT 1", connection_id=6) for _ in range(10))
    )
    assert peak == 2


@pytest.mark.unit
@pytest.mark.asyncio
async def test_out_of_connections_is_retried_once():
    calls = []

    async def execute(sql):
        calls.append(sql)
        return (
            QueryResult.failure(BUSY) if len(calls) == 1 else QueryResult(success=True)
        )

    executor = _executor(execute, busy_retry_delay_seconds=0)
    result = await executor.execute("SELECT 1", connection_id=6)
    assert result.success and len(calls) == 2


@pytest.mark.unit
@pytest.mark.asyncio
async def test_sql_errors_are_not_retried():
    calls = []

    async def execute(sql):
        calls.append(sql)
        return QueryResult.failure("line 1:8: Column 'x' cannot be resolved")

    executor = _executor(execute, busy_retry_delay_seconds=0)
    result = await executor.execute("SELECT x", connection_id=6)
    assert not result.success and len(calls) == 1

"""
deadline.py — Hard deadlines that don't depend on cooperative cancellation.

asyncio.wait_for() cancels the awaited work on timeout and then waits for it
to finish. Work stuck where cancellation is swallowed or deferred (e.g. a
stream to a service whose network vanished) makes wait_for hang far past its
timeout — one stuck agent call froze a whole evaluation run this way.

run_with_deadline() stops waiting at the deadline no matter what: the work is
cancelled best-effort and abandoned, and TimeoutError is raised.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable

import structlog

logger = structlog.get_logger(__name__)


async def run_with_deadline[T](awaitable: Awaitable[T], timeout: float, label: str) -> T:
    """Await `awaitable` for at most `timeout` seconds, then raise TimeoutError."""
    task = asyncio.ensure_future(awaitable)
    try:
        done, _ = await asyncio.wait({task}, timeout=timeout)
    except asyncio.CancelledError:
        task.cancel()
        raise
    if task in done:
        return task.result()

    task.cancel()
    # Retrieve the eventual outcome so it isn't reported as never retrieved
    task.add_done_callback(lambda t: t.cancelled() or t.exception())
    logger.warning("deadline.exceeded_task_abandoned", label=label, timeout_s=timeout)
    raise TimeoutError(f"{label} exceeded {timeout:.0f}s")

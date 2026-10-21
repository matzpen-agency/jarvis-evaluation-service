"""
evaluation_job_store.py — In-process registry of background evaluation runs.

A full dataset evaluation can take hours (the agent is slow and items run
with bounded concurrency), far longer than any HTTP client should keep one
request open. Callers submit a job, get a job_id back immediately, and poll
its status until it completes.

Jobs live in memory: a service restart loses them, and pollers then get 404.
"""

from __future__ import annotations

import asyncio
import uuid
from collections import OrderedDict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

ProgressCallback = Callable[[int, int], None]
JobRunner = Callable[[ProgressCallback], Awaitable[Any]]


@dataclass
class EvaluationJob:
    """State of one background evaluation."""

    job_id: str
    status: str = "running"  # running | completed | failed
    completed_items: int = 0
    total_items: int | None = None
    result: Any = None
    error: str | None = None
    # Results of the questions finished so far (shown while the run is going)
    completed_cases: list[Any] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(tz=UTC))
    finished_at: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class EvaluationJobStore:
    """Starts evaluations as asyncio tasks and tracks their status."""

    def __init__(self, max_finished_jobs: int = 200) -> None:
        self._jobs: OrderedDict[str, EvaluationJob] = OrderedDict()
        self._tasks: dict[str, asyncio.Task] = {}
        self._max_finished = max_finished_jobs

    def submit(self, runner: JobRunner, metadata: dict[str, Any] | None = None) -> EvaluationJob:
        job = EvaluationJob(job_id=str(uuid.uuid4()), metadata=metadata or {})
        self._jobs[job.job_id] = job
        # Keep a strong reference: the event loop only holds weak ones
        self._tasks[job.job_id] = asyncio.create_task(self._run(job, runner))
        self._evict_finished()
        logger.info("evaluation_job.submitted", job_id=job.job_id, **job.metadata)
        return job

    def get(self, job_id: str) -> EvaluationJob | None:
        return self._jobs.get(job_id)

    async def _run(self, job: EvaluationJob, runner: JobRunner) -> None:
        class Progress:
            """Callable progress reporter; .add_case() records a finished question."""

            def __call__(self, completed: int, total: int) -> None:
                job.completed_items = completed
                job.total_items = total

            def add_case(self, case: Any) -> None:
                job.completed_cases.append(case)

        on_progress = Progress()

        try:
            job.result = await runner(on_progress)
            job.status = "completed"
        except Exception as exc:
            job.status = "failed"
            job.error = str(exc) or type(exc).__name__
            logger.error("evaluation_job.failed", job_id=job.job_id, error=job.error, exc_info=True)
        finally:
            job.finished_at = datetime.now(tz=UTC)
            self._tasks.pop(job.job_id, None)
            logger.info("evaluation_job.finished", job_id=job.job_id, status=job.status)

    def _evict_finished(self) -> None:
        finished = [jid for jid, job in self._jobs.items() if job.status != "running"]
        for job_id in finished[: max(0, len(finished) - self._max_finished)]:
            self._jobs.pop(job_id, None)


_job_store = EvaluationJobStore()


def get_job_store() -> EvaluationJobStore:
    return _job_store

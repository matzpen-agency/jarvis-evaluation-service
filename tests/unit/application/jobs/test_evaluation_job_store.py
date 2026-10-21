"""
test_evaluation_job_store.py — Background evaluation jobs.
"""

from __future__ import annotations

import asyncio

import pytest

from src.application.jobs.evaluation_job_store import EvaluationJobStore


@pytest.mark.unit
@pytest.mark.asyncio
async def test_job_completes_with_progress_and_result():
    store = EvaluationJobStore()
    release = asyncio.Event()

    async def runner(progress):
        progress(1, 2)
        await release.wait()
        progress(2, 2)
        return {"ok": True}

    job = store.submit(runner, metadata={"connection_id": 6})
    await asyncio.sleep(0)
    assert store.get(job.job_id).status == "running"
    assert store.get(job.job_id).completed_items == 1

    release.set()
    for _ in range(10):
        await asyncio.sleep(0)
    finished = store.get(job.job_id)
    assert finished.status == "completed"
    assert finished.result == {"ok": True}
    assert (finished.completed_items, finished.total_items) == (2, 2)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_job_failure_is_captured():
    store = EvaluationJobStore()

    async def runner(progress):
        raise RuntimeError("Dataset 'x' not found in Langfuse.")

    job = store.submit(runner)
    for _ in range(5):
        await asyncio.sleep(0)
    assert store.get(job.job_id).status == "failed"
    assert "not found" in store.get(job.job_id).error


@pytest.mark.unit
def test_unknown_job_is_none():
    assert EvaluationJobStore().get("nope") is None


@pytest.mark.unit
@pytest.mark.asyncio
async def test_finished_questions_are_visible_while_running():
    store = EvaluationJobStore()
    release = asyncio.Event()

    async def runner(progress):
        progress.add_case({"question_id": "q1", "scores": {"contains_accuracy": 1.0}})
        progress(1, 3)
        await release.wait()
        return "done"

    job = store.submit(runner)
    for _ in range(3):
        await asyncio.sleep(0)
    running = store.get(job.job_id)
    assert running.status == "running"
    assert (running.completed_items, running.total_items) == (1, 3)
    assert running.completed_cases == [{"question_id": "q1", "scores": {"contains_accuracy": 1.0}}]
    release.set()

"""
test_evaluation_engine.py — Unit tests for EvaluationEngine.
"""

from __future__ import annotations

import pytest

from src.domain.entities.evaluation_result import EvaluationResult
from src.domain.evaluators.base_evaluator import BaseEvaluator
from src.domain.services.evaluation_engine import EvaluationEngine


class DummyEvaluator(BaseEvaluator):
    def __init__(self, name: str, score: float = 1.0, should_raise: bool = False):
        self._name = name
        self._score = score
        self._should_raise = should_raise

    @property
    def name(self) -> str:
        return self._name

    async def evaluate(self, context) -> EvaluationResult:
        if self._should_raise:
            raise RuntimeError("Dummy error")
        return EvaluationResult(
            evaluator_name=self._name,
            score=self._score,
            passed=self._score >= 0.5,
        )


@pytest.mark.unit
def test_initialization_fails_with_empty_list():
    with pytest.raises(ValueError, match="at least one evaluator"):
        EvaluationEngine([])


@pytest.mark.unit
def test_evaluator_names():
    e1 = DummyEvaluator("e1")
    e2 = DummyEvaluator("e2")
    engine = EvaluationEngine([e1, e2])
    assert engine.evaluator_names == ["e1", "e2"]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_run_all_success(sample_context):
    e1 = DummyEvaluator("e1", 0.8)
    e2 = DummyEvaluator("e2", 0.4)
    engine = EvaluationEngine([e1, e2])

    results = await engine.run_all(sample_context)
    assert len(results) == 2
    assert results[0].evaluator_name == "e1"
    assert results[0].score == 0.8
    assert results[0].passed is True
    assert results[1].evaluator_name == "e2"
    assert results[1].score == 0.4
    assert results[1].passed is False


@pytest.mark.unit
@pytest.mark.asyncio
async def test_run_all_with_exception(sample_context):
    e1 = DummyEvaluator("e1", 0.8)
    e2 = DummyEvaluator("e2", should_raise=True)
    engine = EvaluationEngine([e1, e2])

    results = await engine.run_all(sample_context)
    assert len(results) == 2
    assert results[0].evaluator_name == "e1"
    assert results[0].score == 0.8

    assert results[1].evaluator_name == "e2"
    assert results[1].score == 0.0
    assert results[1].passed is False
    assert "Dummy error" in results[1].error

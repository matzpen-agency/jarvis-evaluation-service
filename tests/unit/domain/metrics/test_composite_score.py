"""
test_composite_score.py — Unit tests for CompositeScoreCalculator.
"""

from __future__ import annotations

import pytest

from src.domain.entities.dataset_run import AccuracyStats
from src.domain.metrics.composite_score_calculator import (
    CompositeScoreCalculator,
    CompositeWeights,
)


@pytest.mark.unit
def test_perfect_scores():
    calc = CompositeScoreCalculator()
    stats = AccuracyStats(
        execution_accuracy=1.0,
        contains_accuracy=1.0,
        sql_exact_match=1.0,
        time_shift_score=1.0,
    )
    assert calc.calculate(stats) == 1.0


@pytest.mark.unit
def test_zero_scores():
    calc = CompositeScoreCalculator()
    stats = AccuracyStats()
    assert calc.calculate(stats) == 0.0


@pytest.mark.unit
def test_weighted_computation():
    weights = CompositeWeights(
        execution_accuracy=0.5,
        contains_accuracy=0.2,
        sql_exact_match=0.2,
        time_shift=0.1,
    )
    calc = CompositeScoreCalculator(weights)
    stats = AccuracyStats(
        execution_accuracy=1.0,
        contains_accuracy=0.0,
        sql_exact_match=0.0,
        time_shift_score=0.0,
    )
    assert calc.calculate(stats) == pytest.approx(0.5, abs=0.001)


@pytest.mark.unit
def test_invalid_weights_raises():
    with pytest.raises(ValueError, match=r"sum to 1\.0"):
        weights = CompositeWeights(
            execution_accuracy=0.99,
            contains_accuracy=0.0,
            sql_exact_match=0.0,
            time_shift=0.0,
        )
        weights.validate()

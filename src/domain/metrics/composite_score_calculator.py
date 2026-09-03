"""
composite_score_calculator.py — Weighted composite score from evaluator scores.

Weights are configurable via Settings and must sum to 1.0.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.domain.entities.dataset_run import AccuracyStats
from src.domain.metrics.base_metric_calculator import BaseMetricCalculator


@dataclass
class CompositeWeights:
    """Configurable weights for the composite score formula."""

    execution_accuracy: float = 0.60
    contains_accuracy: float = 0.15
    sql_exact_match: float = 0.15
    time_shift: float = 0.10
    component_match: float = 0.0
    schema_hallucination: float = 0.0
    dialect_error: float = 0.0

    def validate(self) -> None:
        """Raise ValueError if weights do not sum to 1.0."""
        total = (
            self.execution_accuracy
            + self.contains_accuracy
            + self.sql_exact_match
            + self.time_shift
            + self.component_match
            + self.schema_hallucination
            + self.dialect_error
        )
        if abs(total - 1.0) > 0.001:
            raise ValueError(
                f"CompositeWeights must sum to 1.0, got {total:.4f}"
            )


class CompositeScoreCalculator(BaseMetricCalculator[AccuracyStats, float]):
    """
    Computes a single composite score from individual evaluator averages.

    Input: AccuracyStats with per-evaluator average scores.
    Output: float composite score (0.0-1.0).
    """

    def __init__(self, weights: CompositeWeights | None = None) -> None:
        self._weights = weights or CompositeWeights()
        self._weights.validate()

    def calculate(self, samples: AccuracyStats) -> float:
        score = (
            self._weights.execution_accuracy * samples.execution_accuracy
            + self._weights.contains_accuracy * samples.contains_accuracy
            + self._weights.sql_exact_match * samples.sql_exact_match
            + self._weights.time_shift * samples.time_shift_score
            + self._weights.component_match * samples.component_match
            + self._weights.schema_hallucination * samples.schema_hallucination
            + self._weights.dialect_error * samples.dialect_error
        )
        return round(score, 4)

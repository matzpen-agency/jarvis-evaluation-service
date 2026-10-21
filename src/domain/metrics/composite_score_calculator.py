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

    execution_accuracy: float = 0.15
    contains_accuracy: float = 0.40
    sql_exact_match: float = 0.0
    time_shift: float = 0.25
    component_match: float = 0.10
    schema_validity: float = 0.05
    dialect_validity: float = 0.05

    def validate(self) -> None:
        """Raise ValueError if weights do not sum to 1.0."""
        total = (
            self.execution_accuracy
            + self.contains_accuracy
            + self.sql_exact_match
            + self.time_shift
            + self.component_match
            + self.schema_validity
            + self.dialect_validity
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
        """
        Weighted sum of the evaluator averages. When samples.evaluated_counts
        is known, metrics that judged no sample at all (e.g. time_shift on a
        dataset without date columns) are left out and the remaining weights
        are renormalised, instead of counting them as 0.
        """
        terms = [
            ("execution_accuracy", self._weights.execution_accuracy, samples.execution_accuracy),
            ("contains_accuracy", self._weights.contains_accuracy, samples.contains_accuracy),
            ("sql_exact_match", self._weights.sql_exact_match, samples.sql_exact_match),
            ("time_shift", self._weights.time_shift, samples.time_shift_score),
            ("component_match", self._weights.component_match, samples.component_match),
            (
                "schema_validity",
                self._weights.schema_validity,
                samples.schema_validity,
            ),
            ("dialect_validity", self._weights.dialect_validity, samples.dialect_validity),
        ]
        counts = samples.evaluated_counts
        if counts:
            terms = [t for t in terms if counts.get(t[0], 0) > 0]
        total_weight = sum(weight for _, weight, _ in terms)
        if total_weight <= 0:
            return 0.0
        score = sum(weight * value for _, weight, value in terms) / total_weight
        return round(score, 4)

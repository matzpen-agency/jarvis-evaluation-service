"""
evaluation_result.py — Result of a single evaluator for one dataset item.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class EvaluationResult:
    """
    Output of one BaseEvaluator.evaluate() call.

    score    — 0.0 to 1.0; higher is better.
    passed   — True when the score meets the evaluator's pass threshold.
    details  — Evaluator-specific breakdown data (e.g., match_percentage,
               shift_scores, similarity_scores).
    error    — Non-None when the evaluator itself raised an exception.
    """

    evaluator_name: str
    score: float  # 0.0 - 1.0
    passed: bool
    details: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    @classmethod
    def from_error(cls, evaluator_name: str, error: str) -> EvaluationResult:
        """Create a failed result due to evaluator exception."""
        return cls(
            evaluator_name=evaluator_name,
            score=0.0,
            passed=False,
            details={},
            error=error,
        )

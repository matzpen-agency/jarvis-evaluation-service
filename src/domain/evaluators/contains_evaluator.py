"""
contains_evaluator.py — Checks whether generated results contain expected results.

Useful when generated SQL returns a broader superset of the expected rows.
Score = match_percentage (0.0-1.0). Passes when all expected rows are present.
"""

from __future__ import annotations

import structlog

from src.domain.entities.evaluation_context import EvaluationContext
from src.domain.entities.evaluation_result import EvaluationResult
from src.domain.evaluators.base_evaluator import BaseEvaluator
from src.domain.evaluators.sql_comparison_utils import (
    _requires_order_by,
    evaluate_contains,
)

logger = structlog.get_logger(__name__)

PASS_THRESHOLD = 1.0  # all expected rows must be present


class ContainsEvaluator(BaseEvaluator):
    """
    Checks whether every expected result row is present in the generated result.

    Returns:
      - score = 1.0 when all expected rows found (full subset)
      - score = overlap / expected when partially matched
      - score = 0.0 when nothing matched or execution failed
    """

    def __init__(self, numeric_tolerance: int = 6) -> None:
        self._numeric_tolerance = numeric_tolerance

    @property
    def name(self) -> str:
        return "contains_accuracy"

    async def evaluate(self, context: EvaluationContext) -> EvaluationResult:
        try:
            return self._evaluate(context)
        except Exception as exc:
            logger.error("contains_evaluator.error", error=str(exc))
            return EvaluationResult.from_error(self.name, str(exc))

    def _evaluate(self, context: EvaluationContext) -> EvaluationResult:
        requires_ordering = _requires_order_by(context.expected_sql)
        score, details = evaluate_contains(
            expected_result=context.expected_result,
            generated_result=context.generated_result,
            numeric_tolerance=self._numeric_tolerance,
            requires_ordering=requires_ordering,
        )
        passed = score >= PASS_THRESHOLD

        logger.debug(
            "contains_evaluator.result",
            dataset_item_id=context.dataset_item.id,
            passed=passed,
            score=score,
        )

        return EvaluationResult(
            evaluator_name=self.name,
            score=score,
            passed=passed,
            details=details,
        )

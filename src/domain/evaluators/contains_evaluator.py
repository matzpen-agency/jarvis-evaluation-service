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
        if context.expected_result is None or not context.expected_result.success:
            return EvaluationResult(
                evaluator_name=self.name,
                score=0.0,
                passed=False,
                details={"reason": "expected_sql_execution_failed"},
            )
        if context.generated_result is None or not context.generated_result.success:
            return EvaluationResult(
                evaluator_name=self.name,
                score=0.0,
                passed=False,
                details={"reason": "generated_sql_execution_failed"},
            )

        expected_rows = context.expected_result.as_normalised_row_tuples(
            self._numeric_tolerance
        )
        generated_rows = context.generated_result.as_normalised_row_tuples(
            self._numeric_tolerance
        )

        # Vacuously true: no expected rows means fully satisfied
        if not expected_rows:
            return EvaluationResult(
                evaluator_name=self.name,
                score=1.0,
                passed=True,
                details={
                    "expected_row_count": 0,
                    "generated_row_count": len(generated_rows),
                    "match_percentage": 1.0,
                    "reason": "no_expected_rows",
                },
            )

        expected_set = frozenset(expected_rows)
        generated_set = frozenset(generated_rows)

        matched = expected_set & generated_set
        match_percentage = len(matched) / len(expected_set)
        passed = match_percentage >= PASS_THRESHOLD

        logger.debug(
            "contains_evaluator.result",
            dataset_item_id=context.dataset_item.id,
            expected_count=len(expected_set),
            matched_count=len(matched),
            match_percentage=match_percentage,
        )

        return EvaluationResult(
            evaluator_name=self.name,
            score=round(match_percentage, 4),
            passed=passed,
            details={
                "expected_row_count": len(expected_set),
                "generated_row_count": len(generated_set),
                "matched_row_count": len(matched),
                "match_percentage": round(match_percentage, 4),
            },
        )

"""
contains_evaluator.py — Checks whether generated results contain expected results.

PRIMARY METRIC. The generated result passes when it contains every expected
column (matched by data, not by name; extra generated columns are allowed)
and its rows match the expected rows 100% — in order when the expected SQL
has an outer ORDER BY. Score is 1.0 or 0.0.
"""

from __future__ import annotations

import structlog

from src.domain.entities.evaluation_context import EvaluationContext
from src.domain.entities.evaluation_result import EvaluationResult
from src.domain.evaluators.base_evaluator import BaseEvaluator
from src.domain.evaluators.sql_comparison_utils import evaluate_execution_match

logger = structlog.get_logger(__name__)


class ContainsEvaluator(BaseEvaluator):
    """
    Checks that the generated result contains the expected result.

    Returns:
      - score = 1.0 when all expected columns are found and all rows match
      - score = 0.0 otherwise (details["reason"] says why)
      - not applicable when the golden SQL itself fails to execute
    """

    def __init__(self, numeric_tolerance: int = 6) -> None:
        self._numeric_tolerance = numeric_tolerance

    @property
    def name(self) -> str:
        return "contains_accuracy"

    async def evaluate(self, context: EvaluationContext) -> EvaluationResult:
        try:
            result = evaluate_execution_match(
                context,
                self.name,
                allow_extra_columns=True,
                numeric_tolerance=self._numeric_tolerance,
            )
        except Exception as exc:
            logger.error("contains_evaluator.error", error=str(exc))
            return EvaluationResult.from_error(self.name, str(exc))

        logger.debug(
            "contains_evaluator.result",
            dataset_item_id=context.dataset_item.id,
            passed=result.passed,
            score=result.score,
        )
        return result

"""
execution_accuracy_evaluator.py — Exact execution accuracy.

The generated result must equal the expected result: same number of columns
(matched by data, so column order and names don't matter), same rows, and the
same row order when the expected SQL has an outer ORDER BY (rows tied on the
sort key may be in any order). Score: 1.0 if results match, 0.0 otherwise.
"""

from __future__ import annotations

import structlog

from src.domain.entities.evaluation_context import EvaluationContext
from src.domain.entities.evaluation_result import EvaluationResult
from src.domain.evaluators.base_evaluator import BaseEvaluator
from src.domain.evaluators.sql_comparison_utils import evaluate_execution_match

logger = structlog.get_logger(__name__)


class ExecutionAccuracyEvaluator(BaseEvaluator):
    """
    Compares expected query result vs generated query result exactly.

    Like contains accuracy, but extra generated columns fail the check.
    Not applicable when the golden SQL fails to execute.
    """

    def __init__(self, numeric_tolerance: int = 6) -> None:
        self._numeric_tolerance = numeric_tolerance

    @property
    def name(self) -> str:
        return "execution_accuracy"

    async def evaluate(self, context: EvaluationContext) -> EvaluationResult:
        try:
            result = evaluate_execution_match(
                context,
                self.name,
                allow_extra_columns=False,
                numeric_tolerance=self._numeric_tolerance,
            )
        except Exception as exc:
            logger.error("execution_accuracy_evaluator.error", error=str(exc))
            return EvaluationResult.from_error(self.name, str(exc))

        logger.debug(
            "execution_accuracy.result",
            dataset_item_id=context.dataset_item.id,
            passed=result.passed,
            details=result.details,
        )
        return result

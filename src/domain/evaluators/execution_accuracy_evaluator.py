"""
execution_accuracy_evaluator.py — Compares expected vs generated query results.

Supports three comparison modes:
  1. Exact equality (ordered)
  2. Order-independent equality (set comparison) — DEFAULT
  3. Numeric-tolerance equality

Score: 1.0 if results match, 0.0 otherwise.
"""

from __future__ import annotations

import structlog

from src.domain.entities.evaluation_context import EvaluationContext
from src.domain.entities.evaluation_result import EvaluationResult
from src.domain.evaluators.base_evaluator import BaseEvaluator

logger = structlog.get_logger(__name__)

PASS_THRESHOLD = 1.0  # exact match required


class ExecutionAccuracyEvaluator(BaseEvaluator):
    """
    Compares expected query result vs generated query result.

    When both Trino executions succeed, row sets are compared.
    If either execution failed, score = 0.0.
    """

    def __init__(self, numeric_tolerance: int = 6) -> None:
        self._numeric_tolerance = numeric_tolerance

    @property
    def name(self) -> str:
        return "execution_accuracy"

    async def evaluate(self, context: EvaluationContext) -> EvaluationResult:
        try:
            return self._evaluate(context)
        except Exception as exc:
            logger.error("execution_accuracy_evaluator.error", error=str(exc))
            return EvaluationResult.from_error(self.name, str(exc))

    def _evaluate(self, context: EvaluationContext) -> EvaluationResult:
        # ── Guard: execution must have succeeded ──────────────────────────────
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

        # Order-independent set comparison
        expected_set = frozenset(expected_rows)
        generated_set = frozenset(generated_rows)

        exact_ordered = expected_rows == generated_rows
        order_independent = expected_set == generated_set

        score = 1.0 if order_independent else 0.0
        passed = score >= PASS_THRESHOLD

        logger.debug(
            "execution_accuracy.result",
            dataset_item_id=context.dataset_item.id,
            expected_rows=len(expected_rows),
            generated_rows=len(generated_rows),
            exact_ordered=exact_ordered,
            order_independent=order_independent,
        )

        return EvaluationResult(
            evaluator_name=self.name,
            score=score,
            passed=passed,
            details={
                "expected_row_count": len(expected_rows),
                "generated_row_count": len(generated_rows),
                "exact_ordered_match": exact_ordered,
                "order_independent_match": order_independent,
            },
        )

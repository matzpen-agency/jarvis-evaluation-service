"""
execution_accuracy_evaluator.py — Compares expected vs generated query results.

Supports three comparison modes:
  1. Exact equality (ordered)
  2. Order-independent equality (set comparison) — DEFAULT
  3. Numeric-tolerance equality

Score: 1.0 if results match, 0.0 otherwise.

Both expected and generated result rows are normalized via _sort_dataframe
(column-order-invariant sort) before comparison.
"""

from __future__ import annotations

from collections import Counter

import structlog

from src.domain.entities.evaluation_context import EvaluationContext
from src.domain.entities.evaluation_result import EvaluationResult
from src.domain.entities.query_result import QueryResult
from src.domain.evaluators.base_evaluator import BaseEvaluator
from src.domain.evaluators.sql_comparison_utils import _sort_dataframe

logger = structlog.get_logger(__name__)

PASS_THRESHOLD = 1.0  # exact match required


class ExecutionAccuracyEvaluator(BaseEvaluator):
    """
    Compares expected query result vs generated query result.

    When both Trino executions succeed, row sets are compared after normalizing
    column order via _sort_dataframe.
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

        # Sort columns and rows for column-order-invariant comparison
        exp_rows, exp_cols = _sort_dataframe(
            context.expected_result.rows, context.expected_result.columns
        )
        gen_rows, gen_cols = _sort_dataframe(
            context.generated_result.rows, context.generated_result.columns
        )

        # Build temporary QueryResults for normalised tuple conversion
        exp_qr = QueryResult(success=True, rows=exp_rows, columns=exp_cols)
        gen_qr = QueryResult(success=True, rows=gen_rows, columns=gen_cols)

        expected_tuples = exp_qr.as_normalised_row_tuples(self._numeric_tolerance)
        generated_tuples = gen_qr.as_normalised_row_tuples(self._numeric_tolerance)

        col_count_match = len(exp_cols) == len(gen_cols)
        row_count_match = len(expected_tuples) == len(generated_tuples)
        order_independent = Counter(expected_tuples) == Counter(generated_tuples)
        exact_ordered = expected_tuples == generated_tuples

        passed = col_count_match and row_count_match and order_independent
        score = 1.0 if passed else 0.0

        logger.debug(
            "execution_accuracy.result",
            dataset_item_id=context.dataset_item.id,
            expected_rows=len(expected_tuples),
            generated_rows=len(generated_tuples),
            exact_ordered=exact_ordered,
            col_count_match=col_count_match,
        )

        return EvaluationResult(
            evaluator_name=self.name,
            score=score,
            passed=passed,
            details={
                "expected_row_count": len(expected_tuples),
                "generated_row_count": len(generated_tuples),
                "exact_ordered_match": exact_ordered,
                "order_independent_match": order_independent,
                "col_count_match": col_count_match,
            },
        )

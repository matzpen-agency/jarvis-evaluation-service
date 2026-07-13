"""
contains_evaluator.py — Checks whether generated results contain expected results.

Useful when generated SQL returns a broader superset of the expected rows.
Score = match_percentage (0.0-1.0). Passes when all expected rows are present.
"""

from __future__ import annotations

import structlog

from src.domain.entities.evaluation_context import EvaluationContext
from src.domain.entities.evaluation_result import EvaluationResult
from src.domain.entities.query_result import QueryResult
from src.domain.evaluators.base_evaluator import BaseEvaluator
from collections import Counter

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

        n_expected_cols = len(context.expected_result.columns)
        n_generated_cols = len(context.generated_result.columns)

        # Generated result must have at least as many columns as expected.
        # We do NOT match by column name because Trino often uses auto-generated
        # names like '_col0' for aggregations in one execution and 'avg(quantity)'
        # in another, even when the SQL is identical.  Positional alignment is the
        # only reliable strategy.
        expected_cols = [c.lower().strip() for c in context.expected_result.columns]
        generated_cols = [c.lower().strip() for c in context.generated_result.columns]

        # Map expected columns to generated columns:
        # 1. Match by name first (supports reordered and extra columns)
        # 2. Fall back to positional alignment for columns that don't match by name
        #    (handles Trino auto-generated names like '_col0' vs 'avg(quantity)')
        col_indices: list[int | None] = [None] * n_expected_cols
        used_gen_indices: set[int] = set()

        for i, exp_col in enumerate(expected_cols):
            if exp_col in generated_cols:
                idx = generated_cols.index(exp_col)
                col_indices[i] = idx
                used_gen_indices.add(idx)

        gen_idx = 0
        for i in range(n_expected_cols):
            if col_indices[i] is None:
                while gen_idx < n_generated_cols and gen_idx in used_gen_indices:
                    gen_idx += 1
                if gen_idx < n_generated_cols:
                    col_indices[i] = gen_idx
                    used_gen_indices.add(gen_idx)
                    gen_idx += 1
                else:
                    return EvaluationResult(
                        evaluator_name=self.name,
                        score=0.0,
                        passed=False,
                        details={
                            "reason": "missing_expected_columns",
                            "expected_columns": expected_cols,
                            "generated_columns": generated_cols,
                        },
                    )

        # Use the shared normaliser on expected rows (all columns)
        expected_rows = context.expected_result.as_normalised_row_tuples(
            self._numeric_tolerance
        )

        # Extract only matching columns from the generated rows
        projected_gen_rows = []
        for row in context.generated_result.rows:
            # Handle row index bounds checking safely using mapped indices
            if all(idx < len(row) for idx in col_indices):
                projected_gen_rows.append([row[idx] for idx in col_indices])
            else:
                return EvaluationResult(
                    evaluator_name=self.name,
                    score=0.0,
                    passed=False,
                    details={"reason": "generated_row_too_short"},
                )

        # Normalise projected rows using the shared QueryResult helper
        tmp_result = QueryResult(success=True, rows=projected_gen_rows, columns=context.expected_result.columns)
        generated_rows = tmp_result.as_normalised_row_tuples(self._numeric_tolerance)

        # Verify the number of rows must be identical
        if len(expected_rows) != len(generated_rows):
            return EvaluationResult(
                evaluator_name=self.name,
                score=0.0,
                passed=False,
                details={
                    "reason": "row_count_mismatch",
                    "expected_row_count": len(expected_rows),
                    "generated_row_count": len(generated_rows),
                },
            )

        # Every expected value must match (order-independent)
        expected_counter = Counter(expected_rows)
        generated_counter = Counter(generated_rows)
        passed = all(generated_counter[row] >= count for row, count in expected_counter.items())
        score = 1.0 if passed else 0.0

        logger.debug(
            "contains_evaluator.result",
            dataset_item_id=context.dataset_item.id,
            expected_count=len(expected_rows),
            generated_count=len(generated_rows),
            passed=passed,
        )

        return EvaluationResult(
            evaluator_name=self.name,
            score=score,
            passed=passed,
            details={
                "expected_row_count": len(expected_rows),
                "generated_row_count": len(generated_rows),
                "passed": passed,
            },
        )

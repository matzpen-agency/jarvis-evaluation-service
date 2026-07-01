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
from collections import Counter
from typing import Any
from typing import Any

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

        expected_cols = [c.lower().strip() for c in context.expected_result.columns]
        generated_cols = [c.lower().strip() for c in context.generated_result.columns]

        # Verify every expected column exists in generated result
        if not all(col in generated_cols for col in expected_cols):
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

        # Map expected columns to generated column indices
        col_indices = [generated_cols.index(col) for col in expected_cols]

        expected_rows = context.expected_result.as_normalised_row_tuples(
            self._numeric_tolerance
        )

        # Extract only matching columns from the generated rows
        projected_gen_rows = []
        for row in context.generated_result.rows:
            # Handle row index bounds checking safely
            if len(row) > max(col_indices):
                projected_gen_rows.append([row[idx] for idx in col_indices])
            else:
                return EvaluationResult(
                    evaluator_name=self.name,
                    score=0.0,
                    passed=False,
                    details={"reason": "generated_row_length_mismatch"},
                )

        # Normalise projected rows using helper
        def _normalise_rows(rows: list[list[Any]], numeric_tolerance: int) -> list[tuple]:
            normalised = []
            for row in rows:
                norm_row = []
                for cell in row:
                    if cell is None:
                        norm_row.append("")
                    elif isinstance(cell, str):
                        stripped = cell.strip()
                        try:
                            norm_row.append(round(float(stripped), numeric_tolerance))
                        except ValueError:
                            norm_row.append(stripped.lower())
                    elif isinstance(cell, (float, int)):
                        norm_row.append(round(float(cell), numeric_tolerance))
                    else:
                        norm_row.append(str(cell).strip().lower())
                normalised.append(tuple(norm_row))
            return normalised

        generated_rows = _normalise_rows(projected_gen_rows, self._numeric_tolerance)

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

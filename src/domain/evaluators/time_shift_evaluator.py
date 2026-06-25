"""
time_shift_evaluator.py — Evaluates SQL correctness across multiple date offsets.

For each configured time shift (e.g. -1, -7, -14, -30, -60 days):
  1. Inject date offset into both expected and generated SQL by replacing
     CURRENT_DATE with a Trino date arithmetic expression.
  2. Re-execute both queries via the QueryExecutor.
  3. Compare results using the same logic as ExecutionAccuracyEvaluator.

Aggregate score = mean of all individual shift scores.

Purpose: Detect queries that accidentally only work for a specific point in time.
"""

from __future__ import annotations

import asyncio
import re

import structlog

from src.domain.entities.evaluation_context import EvaluationContext
from src.domain.entities.evaluation_result import EvaluationResult
from src.domain.evaluators.base_evaluator import BaseEvaluator
from src.ports.query_executor import QueryExecutor

logger = structlog.get_logger(__name__)

# Trino date arithmetic: DATE_ADD('day', -N, CURRENT_DATE)
_DATE_PATTERN = re.compile(r"\bCURRENT_DATE\b", re.IGNORECASE)


def _inject_date_offset(sql: str, offset_days: int) -> str:
    """
    Replace CURRENT_DATE with DATE_ADD('day', <offset>, CURRENT_DATE).
    offset_days is typically negative (past).
    """
    if not _DATE_PATTERN.search(sql):
        return sql
    replacement = f"DATE_ADD('day', {offset_days}, CURRENT_DATE)"
    return _DATE_PATTERN.sub(replacement, sql)


class TimeShiftEvaluator(BaseEvaluator):
    """
    Runs the same query under multiple date offsets and checks whether
    results remain consistent (i.e., the SQL logic is time-agnostic).

    Requires a QueryExecutor for re-executing shifted queries.
    """

    def __init__(
        self,
        query_executor: QueryExecutor,
        offsets_days: list[int] | None = None,
        numeric_tolerance: int = 6,
    ) -> None:
        self._executor = query_executor
        self._offsets = offsets_days if offsets_days is not None else [-1, -7, -14, -30, -60]
        self._numeric_tolerance = numeric_tolerance

    @property
    def name(self) -> str:
        return "time_shift"

    async def evaluate(self, context: EvaluationContext) -> EvaluationResult:
        try:
            return await self._evaluate(context)
        except Exception as exc:
            logger.error("time_shift_evaluator.error", error=str(exc))
            return EvaluationResult.from_error(self.name, str(exc))

    async def _evaluate(self, context: EvaluationContext) -> EvaluationResult:
        if not context.expected_sql or not context.generated_sql:
            return EvaluationResult(
                evaluator_name=self.name,
                score=0.0,
                passed=False,
                details={"reason": "missing_sql"},
            )

        shift_results: list[dict] = []
        scores: list[float] = []

        # Run all shifts concurrently
        tasks = [
            self._evaluate_shift(context, offset)
            for offset in self._offsets
        ]
        shift_outcomes = await asyncio.gather(*tasks, return_exceptions=True)

        for offset, outcome in zip(self._offsets, shift_outcomes, strict=True):
            if isinstance(outcome, dict):
                shift_results.append(outcome)
                scores.append(outcome["score"])
            else:
                logger.warning(
                    "time_shift.shift_failed",
                    offset_days=offset,
                    error=str(outcome),
                )
                shift_results.append({"offset_days": offset, "score": 0.0, "error": str(outcome)})
                scores.append(0.0)

        aggregate_score = round(sum(scores) / len(scores), 4) if scores else 0.0
        passed = aggregate_score >= 0.8  # 80% of shifts must pass

        logger.debug(
            "time_shift.aggregate",
            dataset_item_id=context.dataset_item.id,
            aggregate_score=aggregate_score,
            shift_scores=scores,
        )

        return EvaluationResult(
            evaluator_name=self.name,
            score=aggregate_score,
            passed=passed,
            details={
                "aggregate_score": aggregate_score,
                "offsets_days": self._offsets,
                "shift_results": shift_results,
            },
        )

    async def _evaluate_shift(self, context: EvaluationContext, offset_days: int) -> dict:
        """Run one date-shifted comparison and return a result dict."""
        expected_shifted = _inject_date_offset(context.expected_sql, offset_days)
        generated_shifted = _inject_date_offset(context.generated_sql, offset_days)

        expected_result, generated_result = await asyncio.gather(
            self._executor.execute(expected_shifted),
            self._executor.execute(generated_shifted),
        )

        if not expected_result.success or not generated_result.success:
            return {
                "offset_days": offset_days,
                "score": 0.0,
                "expected_error": expected_result.error,
                "generated_error": generated_result.error,
            }

        expected_rows = frozenset(
            expected_result.as_normalised_row_tuples(self._numeric_tolerance)
        )
        generated_rows = frozenset(
            generated_result.as_normalised_row_tuples(self._numeric_tolerance)
        )

        score = 1.0 if expected_rows == generated_rows else 0.0
        return {
            "offset_days": offset_days,
            "score": score,
            "expected_row_count": len(expected_rows),
            "generated_row_count": len(generated_rows),
        }

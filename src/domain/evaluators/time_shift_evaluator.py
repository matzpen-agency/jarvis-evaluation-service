"""
time_shift_evaluator.py — Evaluates SQL correctness across multiple date offsets.

For each configured time shift (e.g. -1, -7, -14, -30, -60 days):
  1. Wrap both expected and generated SQL with Trino CTEs that shift ALL
     date/timestamp columns in the referenced tables using date_add().
  2. Re-execute both shifted queries via the QueryExecutor.
  3. Compare results using evaluate_contains (column-order invariant & fractional containment).

Aggregate score = mean of all individual shift scores.
"""

from __future__ import annotations

import asyncio

import structlog

from src.domain.entities.evaluation_context import EvaluationContext
from src.domain.entities.evaluation_result import EvaluationResult
from src.domain.evaluators.base_evaluator import BaseEvaluator
from src.domain.evaluators.sql_comparison_utils import (
    dynamically_wrap_with_yaml_cte,
    evaluate_contains,
)
from src.ports.query_executor import QueryExecutor

logger = structlog.get_logger(__name__)


class TimeShiftEvaluator(BaseEvaluator):
    """
    Runs the same query under multiple date offsets and checks whether
    results remain consistent (i.e. the SQL logic is time-agnostic).

    Uses _dynamically_wrap_with_yaml_cte to shift literal date/timestamp
    columns in the underlying tables, covering all temporal representations.
    Delegates result comparison to evaluate_contains.
    """

    def __init__(
        self,
        query_executor: QueryExecutor,
        offsets_days: list[int] | None = None,
        numeric_tolerance: int = 6,
        table_resolver=None,
    ) -> None:
        self._executor = query_executor
        self._offsets = offsets_days if offsets_days is not None else [-1, -7, -14, -30, -60]
        self._numeric_tolerance = numeric_tolerance
        self._resolver = table_resolver

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

        # Build schema map once for all shifts
        schema_map: dict[str, dict[str, str]] = {}
        if self._resolver is not None:
            try:
                schema_map = await self._resolver.get_table_schema_map()
            except Exception as exc:
                logger.warning(
                    "time_shift_evaluator.schema_map_failed", error=str(exc)
                )

        shift_results: list[dict] = []
        scores: list[float] = []

        # Run all shifts concurrently
        tasks = [
            self._evaluate_shift(context, offset, schema_map)
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
                shift_results.append(
                    {"offset_days": offset, "score": 0.0, "error": str(outcome)}
                )
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

    async def _evaluate_shift(
        self,
        context: EvaluationContext,
        offset_days: int,
        schema_map: dict[str, dict[str, str]],
    ) -> dict:
        """
        Wrap both SQLs with date-shifting CTEs, execute, and compare with evaluate_contains.
        """
        expected_shifted = dynamically_wrap_with_yaml_cte(
            context.expected_sql, offset_days, schema_map
        )
        generated_shifted = dynamically_wrap_with_yaml_cte(
            context.generated_sql, offset_days, schema_map
        )

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

        score, details = evaluate_contains(
            expected_result=expected_result,
            generated_result=generated_result,
            numeric_tolerance=self._numeric_tolerance,
        )

        return {
            "offset_days": offset_days,
            "score": score,
            "passed": score >= 1.0,
            **details,
        }

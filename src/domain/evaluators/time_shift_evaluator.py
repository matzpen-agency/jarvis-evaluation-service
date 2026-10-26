"""
time_shift_evaluator.py — Checks that a contains-match is not a coincidence.

A generated query can match the expected result today by accident — e.g. it
hard-codes '2026-09-01' where the golden SQL says "last month", or it reads a
different date column that happens to give the same rows on this snapshot.

For each configured offset (e.g. -1, -7, -14, -30, -60 days):
  1. Wrap both expected and (executed) generated SQL with CTEs that shift
     every date/timestamp column of the referenced tables by the offset,
     rendered in the connection's dialect.
  2. Re-execute both where they first ran: the golden SQL on the item's
     connection, the generated SQL on the agent's engine when it ran there.
  3. Compare with the same contains check as the primary metric.

Score = passing shifts / conclusive shifts. A shift is inconclusive (not
counted) when the shifted golden SQL fails or both results are truncated.
Passed only when every conclusive shift passes. Not applicable when the
referenced tables have no temporal columns — shifting would change nothing.
"""

from __future__ import annotations

import asyncio
from typing import Any

import structlog

from src.domain.entities.evaluation_context import EvaluationContext
from src.domain.entities.evaluation_result import EvaluationResult
from src.domain.evaluators.base_evaluator import BaseEvaluator
from src.domain.evaluators.sql_comparison_utils import (
    compare_results,
    compare_with_order_policy,
    dynamically_wrap_with_yaml_cte,
    has_no_data,
)
from src.domain.services.deadline import run_with_deadline
from src.ports.order_judge import OrderJudge
from src.ports.query_executor import QueryExecutor

logger = structlog.get_logger(__name__)


class TimeShiftEvaluator(BaseEvaluator):
    """
    Runs both queries under multiple date offsets and checks that the
    generated result keeps containing the expected result.
    """

    def __init__(
        self,
        query_executor: QueryExecutor,
        offsets_days: list[int] | None = None,
        numeric_tolerance: int = 6,
        table_resolver=None,
        shift_timeout_seconds: float = 180.0,
        order_judge: OrderJudge | None = None,
        agent_executor: QueryExecutor | None = None,
    ) -> None:
        self._executor = query_executor
        self._agent_executor = agent_executor
        self._offsets = offsets_days if offsets_days is not None else [-1, -7, -14, -30, -60]
        self._numeric_tolerance = numeric_tolerance
        self._resolver = table_resolver
        self._shift_timeout = shift_timeout_seconds
        self._order_judge = order_judge

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
        if not context.expected_sql:
            return EvaluationResult.not_applicable(self.name, "missing_sql")
        if not context.generated_sql:
            return EvaluationResult(
                evaluator_name=self.name,
                score=0.0,
                passed=False,
                details={"reason": "missing_sql"},
            )
        if context.expected_result is not None and not context.expected_result.success:
            return EvaluationResult.not_applicable(
                self.name, "expected_sql_execution_failed"
            )

        if has_no_data(context.expected_result):
            return EvaluationResult.not_applicable(self.name, "golden_result_has_no_data")

        schema_map = await self._schema_map(context)
        expected_sql = context.executed_expected_sql or context.expected_sql
        generated_sql = context.executed_generated_sql or context.generated_sql
        dialect = context.dialect or "trino"
        # Generated SQL on the agent's engine stays in the agent's dialect
        generated_dialect = context.agent_dialect if context.generated_on_agent_engine else dialect

        shifted_pairs = []
        for offset in self._offsets:
            expected_shifted = dynamically_wrap_with_yaml_cte(
                expected_sql, offset, schema_map, dialect
            )
            generated_shifted = dynamically_wrap_with_yaml_cte(
                generated_sql, offset, schema_map, generated_dialect
            )
            shifted_pairs.append((offset, expected_shifted, generated_shifted))

        if all(
            exp_sql == expected_sql and gen_sql == generated_sql
            for _, exp_sql, gen_sql in shifted_pairs
        ):
            return EvaluationResult.not_applicable(
                self.name,
                "no_temporal_columns_to_shift",
                schema_available=bool(schema_map),
            )

        outcomes = await asyncio.gather(
            *(
                self._evaluate_shift(
                    context, offset, exp_sql, gen_sql
                )
                for offset, exp_sql, gen_sql in shifted_pairs
            ),
            return_exceptions=True,
        )

        shift_results: list[dict[str, Any]] = []
        for (offset, _, _), outcome in zip(shifted_pairs, outcomes, strict=True):
            if isinstance(outcome, BaseException):
                logger.warning("time_shift.shift_failed", offset_days=offset, error=str(outcome))
                outcome = {
                    "offset_days": offset,
                    "status": "failed",
                    "score": 0.0,
                    "error": str(outcome) or type(outcome).__name__,
                }
            shift_results.append(outcome)

        conclusive = [r for r in shift_results if r["status"] != "inconclusive"]
        details: dict[str, Any] = {
            "offsets_days": self._offsets,
            "shift_results": shift_results,
            "conclusive_shifts": len(conclusive),
            # False when no shift changed the golden result: the question is
            # time-independent on this data, so a pass proves less
            "discriminative": any(r.get("expected_changed") for r in shift_results),
        }
        if not conclusive:
            return EvaluationResult.not_applicable(
                self.name, "no_conclusive_shift", **details
            )

        passed_count = sum(1 for r in conclusive if r["status"] == "passed")
        score = round(passed_count / len(conclusive), 4)
        details["aggregate_score"] = score

        logger.debug(
            "time_shift.aggregate",
            dataset_item_id=context.dataset_item.id,
            aggregate_score=score,
            statuses=[r["status"] for r in shift_results],
        )
        return EvaluationResult(
            evaluator_name=self.name,
            score=score,
            passed=passed_count == len(conclusive),
            details=details,
        )

    async def _schema_map(self, context: EvaluationContext) -> dict[str, dict[str, str]]:
        if context.schema_map:
            return context.schema_map
        if self._resolver is None:
            return {}
        try:
            return await self._resolver.get_table_schema_map(context.connection_id)
        except Exception as exc:
            logger.warning("time_shift_evaluator.schema_map_failed", error=str(exc))
            return {}

    def _generated_executor(self, context: EvaluationContext) -> QueryExecutor:
        if context.generated_on_agent_engine and self._agent_executor is not None:
            return self._agent_executor
        return self._executor

    async def _evaluate_shift(
        self,
        context: EvaluationContext,
        offset_days: int,
        expected_sql: str,
        generated_sql: str,
    ) -> dict[str, Any]:
        """Execute one shifted pair and compare with the contains check."""
        try:
            expected_result, generated_result = await run_with_deadline(
                asyncio.gather(
                    self._executor.execute(expected_sql, connection_id=context.connection_id),
                    self._generated_executor(context).execute(
                        generated_sql, connection_id=context.connection_id
                    ),
                ),
                self._shift_timeout,
                label=f"time shift {offset_days}d",
            )
        except TimeoutError:
            return {
                "offset_days": offset_days,
                "status": "inconclusive",
                "reason": f"shift_timed_out_after_{int(self._shift_timeout)}s",
            }

        if not expected_result.success:
            # Our rewrite broke the golden SQL — says nothing about the agent
            return {
                "offset_days": offset_days,
                "status": "inconclusive",
                "reason": "shifted_expected_sql_failed",
                "expected_error": expected_result.error,
            }
        if not generated_result.success:
            return {
                "offset_days": offset_days,
                "status": "failed",
                "score": 0.0,
                "reason": "shifted_generated_sql_failed",
                "generated_error": generated_result.error,
            }

        # Same order rule as contains accuracy (the judge's decision is cached)
        score, details = await compare_with_order_policy(
            context,
            expected_result,
            generated_result,
            allow_extra_columns=True,
            numeric_tolerance=self._numeric_tolerance,
            order_judge=self._order_judge,
            expected_sql=expected_sql,
        )
        if details.get("inconclusive"):
            status = "inconclusive"
        else:
            status = "passed" if score >= 1.0 else "failed"

        expected_changed = None
        if context.expected_result is not None and context.expected_result.success:
            unchanged, _ = compare_results(
                context.expected_result,
                expected_result,
                allow_extra_columns=False,
                numeric_tolerance=self._numeric_tolerance,
            )
            expected_changed = unchanged < 1.0

        return {
            "offset_days": offset_days,
            "status": status,
            "score": score,
            "expected_changed": expected_changed,
            **details,
        }

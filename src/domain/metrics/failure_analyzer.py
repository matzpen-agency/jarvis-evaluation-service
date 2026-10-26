"""
failure_analyzer.py — Categorises and counts failures across a dataset run.

Takes a list of EvaluationContexts and produces a FailureAnalysis
with detailed breakdowns by category, rate, and timestamp.
"""

from __future__ import annotations

from datetime import UTC, datetime

from src.domain.entities.dataset_run import FailureAnalysis, FailureCategory
from src.domain.entities.evaluation_context import EvaluationContext
from src.domain.metrics.base_metric_calculator import BaseMetricCalculator


class FailureAnalyzer(BaseMetricCalculator[list[EvaluationContext], FailureAnalysis]):
    """
    Analyses all EvaluationContexts and categorises failures.

    Failure categories (mutually exclusive, in priority order):
      1. agent_crash        — agent_crashed=True
      2. timeout            — timed_out=True
      3. sql_execution      — agent produced no SQL
      4. trino_failure      — Trino execution error
      5. validation         — evaluator-detected failure (not execution)
    """

    def calculate(self, samples: list[EvaluationContext]) -> FailureAnalysis:
        if not samples:
            return FailureAnalysis()

        total = len(samples)
        now = datetime.now(tz=UTC)

        agent_crash: list[datetime] = []
        timeout: list[datetime] = []
        sql_exec: list[datetime] = []
        trino_fail: list[datetime] = []
        validation_fail: list[datetime] = []

        for ctx in samples:
            ts = now  # In a full impl, EvaluationContext would carry a timestamp
            if ctx.agent_crashed:
                agent_crash.append(ts)
            elif ctx.timed_out:
                timeout.append(ts)
            elif ctx.agent_response is None or not ctx.agent_response.succeeded:
                sql_exec.append(ts)
            elif (
                (ctx.expected_result and not ctx.expected_result.success)
                or (ctx.generated_result and not ctx.generated_result.success)
            ):
                trino_fail.append(ts)
            elif ctx.failed:
                validation_fail.append(ts)

        all_failures = (
            agent_crash + timeout + sql_exec + trino_fail + validation_fail
        )
        total_failures = len(all_failures)
        failure_rate = round(total_failures / total, 4) if total > 0 else 0.0

        def _rate(lst: list) -> float:
            return round(len(lst) / total, 4) if total > 0 else 0.0

        categories: list[FailureCategory] = []
        for label, lst in [
            ("agent_crash", agent_crash),
            ("timeout", timeout),
            ("sql_execution_failure", sql_exec),
            ("trino_failure", trino_fail),
            ("validation_failure", validation_fail),
        ]:
            if lst:
                categories.append(
                    FailureCategory(
                        category=label,
                        count=len(lst),
                        rate=_rate(lst),
                        timestamps=lst,
                    )
                )

        return FailureAnalysis(
            total_failures=total_failures,
            failure_rate=failure_rate,
            agent_crash_count=len(agent_crash),
            agent_crash_rate=_rate(agent_crash),
            sql_execution_failure_count=len(sql_exec),
            sql_execution_failure_rate=_rate(sql_exec),
            trino_failure_count=len(trino_fail),
            trino_failure_rate=_rate(trino_fail),
            timeout_count=len(timeout),
            timeout_rate=_rate(timeout),
            validation_failure_count=len(validation_fail),
            validation_failure_rate=_rate(validation_fail),
            categories=categories,
            failure_timestamps=all_failures,
        )

"""
trace_reporter.py — Abstract interface for publishing evaluation traces/results.

Implementations: LangfuseReporter
Future:         DatadogReporter, ArizeReporter, OpenTelemetryReporter, ...
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from src.domain.entities.dataset_run import DatasetRun
from src.domain.entities.evaluation_context import EvaluationContext
from src.domain.entities.evaluation_result import EvaluationResult


class TraceReporter(ABC):
    """
    Generic interface for publishing evaluation traces, spans, and metrics
    to an observability/tracing backend.
    """

    @abstractmethod
    async def create_dataset_run_trace(
        self,
        dataset_name: str,
        run_id: str,
        metadata: dict,
    ) -> str:
        """
        Create a top-level trace for the dataset evaluation run.

        Returns:
            trace_id string for linking child spans.
        """
        ...

    @abstractmethod
    async def report_sample(
        self,
        context: EvaluationContext,
        results: list[EvaluationResult],
        parent_trace_id: str,
        dataset_item_id: str,
        run_name: str,
    ) -> None:
        """
        Report a single sample's evaluation: agent span, SQL executions,
        evaluator scores, and all metadata.
        """
        ...

    @abstractmethod
    async def report_run_summary(
        self,
        run: DatasetRun,
        trace_id: str,
    ) -> None:
        """Write aggregate run metrics back to the reporting backend."""
        ...

    @abstractmethod
    async def flush(self) -> None:
        """Flush all pending events to the backend."""
        ...

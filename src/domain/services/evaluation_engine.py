"""
evaluation_engine.py — Runs all registered evaluator plugins for one sample.

The EvaluationEngine is the plugin registry. It holds a list of BaseEvaluator
instances and executes them concurrently for every EvaluationContext.

To add a new evaluator: instantiate it and add to the list in container.py.
No other code changes required (Open/Closed Principle).
"""

from __future__ import annotations

import asyncio

import structlog

from src.domain.entities.evaluation_context import EvaluationContext
from src.domain.entities.evaluation_result import EvaluationResult
from src.domain.evaluators.base_evaluator import BaseEvaluator

logger = structlog.get_logger(__name__)


class EvaluationEngine:
    """
    Plugin-based evaluator runner.

    Executes all registered evaluators concurrently via asyncio.gather(),
    captures any individual evaluator failures gracefully, and returns
    a complete list of EvaluationResults.
    """

    def __init__(self, evaluators: list[BaseEvaluator]) -> None:
        if not evaluators:
            raise ValueError("EvaluationEngine requires at least one evaluator.")
        self._evaluators = evaluators
        logger.info(
            "evaluation_engine.initialized",
            evaluator_names=[e.name for e in evaluators],
        )

    @property
    def evaluator_names(self) -> list[str]:
        return [e.name for e in self._evaluators]

    async def run_all(self, context: EvaluationContext) -> list[EvaluationResult]:
        """
        Run all evaluators concurrently for the given context.

        Individual evaluator exceptions are captured into
        EvaluationResult.from_error() and never propagate up.

        Returns:
            List of EvaluationResults in the same order as registered evaluators.
        """
        tasks = [self._run_one(evaluator, context) for evaluator in self._evaluators]
        results: list[EvaluationResult] = await asyncio.gather(*tasks)

        logger.debug(
            "evaluation_engine.run_all.complete",
            dataset_item_id=context.dataset_item.id,
            scores={r.evaluator_name: r.score for r in results},
        )
        return list(results)

    async def _run_one(
        self,
        evaluator: BaseEvaluator,
        context: EvaluationContext,
    ) -> EvaluationResult:
        """Run a single evaluator and capture any exception."""
        try:
            return await evaluator.evaluate(context)
        except Exception as exc:
            logger.error(
                "evaluation_engine.evaluator_unhandled_exception",
                evaluator=evaluator.name,
                error=str(exc),
                exc_info=True,
            )
            return EvaluationResult.from_error(evaluator.name, str(exc))

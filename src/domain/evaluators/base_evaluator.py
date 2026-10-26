"""
base_evaluator.py — Abstract base class for all evaluator plugins.

New evaluators must:
  1. Subclass BaseEvaluator
  2. Define a unique `name` property
  3. Implement `evaluate(context) -> EvaluationResult`
  4. Register themselves in the DI container (api/dependencies/container.py)

No other orchestration code needs to change — this is the Open/Closed Principle.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from src.domain.entities.evaluation_context import EvaluationContext
from src.domain.entities.evaluation_result import EvaluationResult


class BaseEvaluator(ABC):
    """
    Plugin interface for evaluation logic.

    Each evaluator receives a fully populated EvaluationContext and returns
    a scored EvaluationResult. Evaluators are stateless and independently
    testable.

    Built-in evaluators:
      - ExecutionAccuracyEvaluator
      - ContainsEvaluator
      - SqlExactMatchEvaluator
      - TimeShiftEvaluator

    Future evaluators (no orchestration changes required):
      - LLMJudgeEvaluator
      - SemanticSqlSimilarityEvaluator
      - CostEfficiencyEvaluator
      - QueryComplexityEvaluator
      - BusinessRuleEvaluator
      - SecurityComplianceEvaluator
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique evaluator identifier used for scoring and reporting."""
        ...

    @abstractmethod
    async def evaluate(self, context: EvaluationContext) -> EvaluationResult:
        """
        Evaluate a single sample.

        Must NEVER raise — capture exceptions into EvaluationResult.from_error().

        Args:
            context: Fully populated evaluation snapshot (read-only).

        Returns:
            EvaluationResult with score (0.0-1.0), pass/fail, and details.
        """
        ...

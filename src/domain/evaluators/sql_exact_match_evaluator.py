"""
sql_exact_match_evaluator.py — Normalized SQL text comparison.

Score = 1.0 if normalized SQL strings match, 0.0 otherwise.
"""

from __future__ import annotations

import re

import structlog

from src.domain.entities.evaluation_context import EvaluationContext
from src.domain.entities.evaluation_result import EvaluationResult
from src.domain.evaluators.base_evaluator import BaseEvaluator

logger = structlog.get_logger(__name__)

PASS_THRESHOLD = 1.0  # exact normalized match required


def _normalize_sql(sql: str) -> str:
    """
    Lightly normalize SQL for comparison:
      - Lowercase
      - Remove SQL comments (-- and /* */)
      - Collapse all whitespace to single space
      - Strip trailing semicolons
    """
    sql = sql.lower()
    # Remove block comments
    sql = re.sub(r"/\*.*?\*/", " ", sql, flags=re.DOTALL)
    # Remove line comments
    sql = re.sub(r"--[^\n]*", " ", sql)
    # Collapse whitespace
    sql = re.sub(r"\s+", " ", sql).strip()
    # Strip trailing semicolons
    sql = sql.rstrip(";").strip()
    return sql


class SqlExactMatchEvaluator(BaseEvaluator):
    """
    Compares generated SQL to expected SQL after normalizing whitespace, casing, and comments.
    """

    @property
    def name(self) -> str:
        return "sql_exact_match"

    async def evaluate(self, context: EvaluationContext) -> EvaluationResult:
        try:
            return self._evaluate(context)
        except Exception as exc:
            logger.error("sql_exact_match_evaluator.error", error=str(exc))
            return EvaluationResult.from_error(self.name, str(exc))

    def _evaluate(self, context: EvaluationContext) -> EvaluationResult:
        if not context.expected_sql:
            return EvaluationResult(
                evaluator_name=self.name,
                score=0.0,
                passed=False,
                details={"reason": "no_expected_sql"},
            )
        if not context.generated_sql:
            return EvaluationResult(
                evaluator_name=self.name,
                score=0.0,
                passed=False,
                details={"reason": "no_generated_sql"},
            )

        norm_exp = _normalize_sql(context.expected_sql)
        norm_gen = _normalize_sql(context.generated_sql)
        passed = norm_exp == norm_gen
        score = 1.0 if passed else 0.0

        logger.debug(
            "sql_exact_match.result",
            dataset_item_id=context.dataset_item.id,
            score=score,
            passed=passed,
        )

        return EvaluationResult(
            evaluator_name=self.name,
            score=score,
            passed=passed,
            details={
                "normalized_match": passed,
                "expected_normalized": norm_exp,
                "generated_normalized": norm_gen,
            },
        )

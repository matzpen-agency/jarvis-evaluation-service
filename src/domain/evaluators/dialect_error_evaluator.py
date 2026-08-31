"""
dialect_error_evaluator.py — Evaluates whether the generated SQL complies with the Trino SQL dialect.
"""

from __future__ import annotations

import structlog
import sqlglot

from src.domain.entities.evaluation_context import EvaluationContext
from src.domain.entities.evaluation_result import EvaluationResult
from src.domain.evaluators.base_evaluator import BaseEvaluator

logger = structlog.get_logger(__name__)


class DialectErrorEvaluator(BaseEvaluator):
    """
    Validates generated SQL against the Trino SQL dialect using AST parsing validation
    and checking for dialect-specific execution signatures.
    """

    @property
    def name(self) -> str:
        return "dialect_error"

    async def evaluate(self, context: EvaluationContext) -> EvaluationResult:
        try:
            return self._evaluate(context)
        except Exception as exc:
            logger.error("dialect_error_evaluator.error", error=str(exc))
            return EvaluationResult.from_error(self.name, str(exc))

    def _evaluate(self, context: EvaluationContext) -> EvaluationResult:
        if not context.generated_sql:
            return EvaluationResult(
                evaluator_name=self.name,
                score=0.0,
                passed=False,
                details={"reason": "no_generated_sql"},
            )

        dialect_err: str | None = None

        # 1. Check if the generated SQL parses successfully in Trino dialect using sqlglot
        try:
            sqlglot.parse_one(context.generated_sql, read="trino")
        except Exception as exc:
            dialect_err = f"sqlglot Trino parser error: {exc}"

        # 2. If parsing succeeded, check for Trino-specific syntax runtime error signatures
        if not dialect_err and context.generated_result and not context.generated_result.success:
            err = (context.generated_result.error or "").lower()
            trino_syntax_signatures = [
                "syntax error",
                "mismatched input",
                "lexical error",
                "parse error",
            ]
            if any(sig in err for sig in trino_syntax_signatures):
                dialect_err = f"Trino execution error: {context.generated_result.error}"

        score = 1.0 if dialect_err is None else 0.0
        passed = score == 1.0

        return EvaluationResult(
            evaluator_name=self.name,
            score=score,
            passed=passed,
            details={"error": dialect_err} if dialect_err else {},
        )

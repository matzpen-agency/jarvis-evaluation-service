"""
dialect_validity_evaluator.py — Is the generated SQL valid for its SQL dialect?

A dialect error is SQL the database can't even interpret: it doesn't parse
in the agent's dialect (Trino by default), or the database rejected it with a
syntax / unknown-function / operator-type error (e.g. MySQL backticks or
Postgres-only functions sent to Trino). Missing tables or columns are not
dialect errors — schema_validity covers those — and a query that runs
but returns wrong rows is not either.

Score (1 = good): 1.0 when the SQL is valid for its dialect, 0.0 on a dialect
error (described in details["error"]).
Not applicable when the agent produced no SQL.
"""

from __future__ import annotations

import re

import sqlglot
import structlog
from sqlglot.errors import ParseError, TokenError

from src.domain.entities.evaluation_context import EvaluationContext
from src.domain.entities.evaluation_result import EvaluationResult
from src.domain.evaluators.base_evaluator import BaseEvaluator

logger = structlog.get_logger(__name__)

# Engine error messages that mean "this SQL isn't valid in my dialect"
_DIALECT_ERROR_PATTERNS = re.compile(
    "|".join(
        [
            r"syntax error",
            r"syntax_error",
            r"mismatched input",
            r"extraneous input",
            r"lexical error",
            r"parse error",
            r"function_not_found",
            r"function .* (does not exist|not registered|not found)",
            r"unexpected parameters",
            r"operator does not exist",
            r"no operator matches",
            r"cannot be applied to",
            r"type_mismatch",
        ]
    ),
    re.IGNORECASE,
)


class DialectValidityEvaluator(BaseEvaluator):
    """Validates generated SQL against the agent's dialect and the engine's errors."""

    @property
    def name(self) -> str:
        return "dialect_validity"

    async def evaluate(self, context: EvaluationContext) -> EvaluationResult:
        try:
            return self._evaluate(context)
        except Exception as exc:
            logger.error("dialect_validity_evaluator.error", error=str(exc))
            return EvaluationResult.from_error(self.name, str(exc))

    def _evaluate(self, context: EvaluationContext) -> EvaluationResult:
        if not context.generated_sql:
            return EvaluationResult.not_applicable(self.name, "no_generated_sql")

        details: dict = {
            "agent_dialect": context.agent_dialect,
            "connection_dialect": context.dialect,
            "transpiled_for_connection": bool(
                context.executed_generated_sql
                and context.executed_generated_sql != context.generated_sql
            ),
        }

        dialect_err: str | None = None
        try:
            parsed = sqlglot.parse_one(context.generated_sql, read=context.agent_dialect)
            if parsed is None:
                dialect_err = "empty statement"
        except (ParseError, TokenError) as exc:
            message = re.sub(r"\x1b\[[0-9;]*m", "", str(exc))  # sqlglot underlines with ANSI codes
            dialect_err = f"not valid {context.agent_dialect} SQL: {message}"

        if dialect_err is None and context.generated_result is not None:
            engine_error = context.generated_result.error or ""
            if not context.generated_result.success and _DIALECT_ERROR_PATTERNS.search(
                engine_error
            ):
                dialect_err = f"rejected by the database: {engine_error[:500]}"

        if dialect_err:
            details["error"] = dialect_err
        return EvaluationResult(
            evaluator_name=self.name,
            score=0.0 if dialect_err else 1.0,
            passed=dialect_err is None,
            details=details,
        )

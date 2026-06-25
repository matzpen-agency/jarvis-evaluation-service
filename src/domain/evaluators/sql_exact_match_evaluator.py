"""
sql_exact_match_evaluator.py — Multi-level SQL text and AST comparison.

Three comparison levels:
  1. Raw      (20% weight) — lowercased + whitespace-collapsed exact match
  2. Normalized (30% weight) — strip comments, aliases, semicolons
  3. AST       (50% weight) — sqlglot canonical form comparison

Score = weighted average of all three similarity values.
"""

from __future__ import annotations

import re

import structlog

from src.domain.entities.evaluation_context import EvaluationContext
from src.domain.entities.evaluation_result import EvaluationResult
from src.domain.evaluators.base_evaluator import BaseEvaluator

logger = structlog.get_logger(__name__)

# Weights for each comparison level
_RAW_WEIGHT = 0.20
_NORMALIZED_WEIGHT = 0.30
_AST_WEIGHT = 0.50

PASS_THRESHOLD = 0.60  # composite SQL match to pass


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


def _ast_canonical(sql: str) -> str:
    """
    Return the sqlglot canonical form of a SQL string.
    Falls back to normalized form on parse errors.
    """
    try:
        import sqlglot

        parsed = sqlglot.parse_one(sql, error_level=sqlglot.ErrorLevel.IGNORE)
        if parsed is not None:
            return parsed.sql(pretty=False).lower().strip()
    except Exception:
        pass
    return _normalize_sql(sql)


def _similarity(a: str, b: str) -> float:
    """Binary similarity: 1.0 if strings match, 0.0 otherwise."""
    return 1.0 if a == b else 0.0


class SqlExactMatchEvaluator(BaseEvaluator):
    """
    Compares generated SQL to expected SQL at three levels of strictness.

    Returns a weighted composite score that rewards AST equivalence most
    heavily, allowing trivial formatting differences to still score highly.
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

        expected = context.expected_sql
        generated = context.generated_sql

        # ── Level 1: Raw ──────────────────────────────────────────────────────
        raw_exp = re.sub(r"\s+", " ", expected.lower()).strip()
        raw_gen = re.sub(r"\s+", " ", generated.lower()).strip()
        raw_score = _similarity(raw_exp, raw_gen)

        # ── Level 2: Normalized ───────────────────────────────────────────────
        norm_exp = _normalize_sql(expected)
        norm_gen = _normalize_sql(generated)
        normalized_score = _similarity(norm_exp, norm_gen)

        # ── Level 3: AST ──────────────────────────────────────────────────────
        ast_exp = _ast_canonical(expected)
        ast_gen = _ast_canonical(generated)
        ast_score = _similarity(ast_exp, ast_gen)

        # ── Composite ─────────────────────────────────────────────────────────
        composite = (
            _RAW_WEIGHT * raw_score
            + _NORMALIZED_WEIGHT * normalized_score
            + _AST_WEIGHT * ast_score
        )
        composite = round(composite, 4)
        passed = composite >= PASS_THRESHOLD

        logger.debug(
            "sql_exact_match.result",
            dataset_item_id=context.dataset_item.id,
            raw=raw_score,
            normalized=normalized_score,
            ast=ast_score,
            composite=composite,
        )

        return EvaluationResult(
            evaluator_name=self.name,
            score=composite,
            passed=passed,
            details={
                "raw_score": raw_score,
                "normalized_score": normalized_score,
                "ast_score": ast_score,
                "composite_score": composite,
            },
        )

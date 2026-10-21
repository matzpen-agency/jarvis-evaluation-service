"""
component_match_evaluator.py — Clause-level SQL similarity using sqlglot ASTs.

Each query is broken into components, collected over every SELECT in it
(main query, CTEs, subqueries, UNION branches):

  select   projected expressions (output aliases ignored)
  from     tables read in FROM and JOIN (catalog/schema qualifiers ignored)
  join     JOIN ON conditions, split into AND-ed predicates
  where    WHERE predicates, split into AND-ed predicates
  group    GROUP BY expressions
  having   HAVING predicates, split into AND-ed predicates
  order    ORDER BY expressions with direction
  limit    LIMIT value

Table aliases are resolved to table names (f.salesamount and
factinternetsales.salesamount are the same column) and symmetric comparisons
are canonicalised (a = b equals b = a).

Each clause gets an F1 score (partial credit for partially matching lists).
Clauses that neither query uses are skipped, so a two-clause query isn't
inflated by six "matching" empty clauses. Score = mean clause F1.
"""

from __future__ import annotations

import functools
from collections import Counter
from typing import Any

import sqlglot.expressions as exp
import structlog

from src.domain.entities.evaluation_context import EvaluationContext
from src.domain.entities.evaluation_result import EvaluationResult
from src.domain.evaluators.base_evaluator import BaseEvaluator
from src.domain.evaluators.sql_comparison_utils import node_arg, safe_parse

logger = structlog.get_logger(__name__)

CLAUSES: tuple[str, ...] = (
    "select",
    "from",
    "join",
    "where",
    "group",
    "having",
    "order",
    "limit",
)

_SYMMETRIC_OPERATORS = (exp.EQ, exp.NEQ, exp.NullSafeEQ, exp.NullSafeNEQ)
_MAX_DIFF_ITEMS = 10


def _resolve_output_reference(
    node: exp.Expression,
    projections: list[exp.Expression],
    output_aliases: dict[str, exp.Expression],
) -> exp.Expression:
    """GROUP BY 1 / ORDER BY total → the projected expression they refer to."""
    if isinstance(node, exp.Literal) and node.is_int:
        idx = int(node.this) - 1
        if 0 <= idx < len(projections):
            return projections[idx]
    if isinstance(node, exp.Column) and not node.table:
        return output_aliases.get(node.name.lower(), node)
    return node


def _split_conjuncts(predicate: exp.Expression | None) -> list[exp.Expression]:
    if predicate is None:
        return []
    if isinstance(predicate, exp.Where | exp.Having):
        predicate = predicate.this
    while isinstance(predicate, exp.Paren):
        predicate = predicate.this
    if isinstance(predicate, exp.And):
        return _split_conjuncts(predicate.this) + _split_conjuncts(predicate.expression)
    return [predicate]


class _ComponentExtractor:
    """Extracts normalised clause components from one SQL query."""

    def __init__(self, parsed: exp.Expression, dialect: str | None) -> None:
        self._dialect = dialect
        self._aliases: dict[str, str] = {}
        for table in parsed.find_all(exp.Table):
            if table.alias:
                self._aliases[table.alias.lower()] = table.name.lower()
        self._parsed = parsed

    def extract(self) -> dict[str, Counter[str]]:
        components: dict[str, Counter[str]] = {c: Counter() for c in CLAUSES}
        for select in self._parsed.find_all(exp.Select):
            projections = [p.unalias() for p in select.expressions]
            output_aliases = {
                p.alias.lower(): p.unalias() for p in select.expressions if p.alias
            }
            resolve = functools.partial(
                _resolve_output_reference,
                projections=projections,
                output_aliases=output_aliases,
            )

            for projection in projections:
                components["select"][self._norm(projection)] += 1

            from_clause = node_arg(select, "from")
            if from_clause is not None:
                components["from"][self._source_name(from_clause.this)] += 1

            for join in select.args.get("joins") or []:
                components["from"][self._source_name(join.this)] += 1
                for predicate in _split_conjuncts(join.args.get("on")):
                    components["join"][self._norm(predicate)] += 1

            for predicate in _split_conjuncts(select.args.get("where")):
                components["where"][self._norm(predicate)] += 1

            group = select.args.get("group")
            if group is not None:
                for expression in group.expressions:
                    components["group"][self._norm(resolve(expression))] += 1

            for predicate in _split_conjuncts(select.args.get("having")):
                components["having"][self._norm(predicate)] += 1

            order = select.args.get("order")
            if order is not None:
                for ordered in order.expressions:
                    direction = "desc" if ordered.args.get("desc") else "asc"
                    key = self._norm(resolve(ordered.this))
                    components["order"][f"{key} {direction}"] += 1

            limit = select.args.get("limit")
            if limit is not None:
                value = limit.expression if limit.expression is not None else limit.this
                components["limit"][self._norm(value) if value is not None else ""] += 1

        # ORDER BY / LIMIT of a UNION live on the set operation, not a SELECT
        if isinstance(self._parsed, exp.SetOperation):
            order = self._parsed.args.get("order")
            if order is not None:
                for ordered in order.expressions:
                    direction = "desc" if ordered.args.get("desc") else "asc"
                    components["order"][f"{self._norm(ordered.this)} {direction}"] += 1
        return components

    def _source_name(self, source: exp.Expression | None) -> str:
        if isinstance(source, exp.Table):
            return source.name.lower()
        return "<subquery>"

    def _norm(self, node: exp.Expression) -> str:
        node = node.copy()
        for column in node.find_all(exp.Column):
            table = column.table.lower()
            if table:
                column.set("table", exp.to_identifier(self._aliases.get(table, table)))
            column.set("db", None)
            column.set("catalog", None)
        for operator in list(node.find_all(*_SYMMETRIC_OPERATORS)):
            left = operator.this.sql(dialect=self._dialect).lower()
            right = operator.expression.sql(dialect=self._dialect).lower()
            if left > right:
                this, other = operator.this, operator.expression
                operator.set("this", other)
                operator.set("expression", this)
        return node.sql(dialect=self._dialect).lower().strip()


def _f1(expected: Counter[str], generated: Counter[str]) -> float:
    if not expected and not generated:
        return 1.0
    if not expected or not generated:
        return 0.0
    overlap = sum((expected & generated).values())
    if overlap == 0:
        return 0.0
    precision = overlap / sum(generated.values())
    recall = overlap / sum(expected.values())
    return 2 * precision * recall / (precision + recall)


class ComponentMatchEvaluator(BaseEvaluator):
    """Compares expected vs generated SQL clause by clause."""

    @property
    def name(self) -> str:
        return "component_match"

    async def evaluate(self, context: EvaluationContext) -> EvaluationResult:
        try:
            return self._evaluate(context)
        except Exception as exc:
            logger.error("component_match_evaluator.error", error=str(exc))
            return EvaluationResult.from_error(self.name, str(exc))

    def _evaluate(self, context: EvaluationContext) -> EvaluationResult:
        if not context.expected_sql:
            return EvaluationResult.not_applicable(self.name, "no_expected_sql")
        if not context.generated_sql:
            return EvaluationResult(
                evaluator_name=self.name,
                score=0.0,
                passed=False,
                details={"reason": "no_generated_sql"},
            )

        parsed_exp = safe_parse(context.expected_sql, context.dialect)
        parsed_gen = safe_parse(context.generated_sql, context.agent_dialect, context.dialect)
        if parsed_exp is None:
            return EvaluationResult.not_applicable(self.name, "expected_sql_parsing_failed")
        if parsed_gen is None:
            return EvaluationResult(
                evaluator_name=self.name,
                score=0.0,
                passed=False,
                details={"reason": "sql_parsing_failed"},
            )

        # Render both in one dialect so equivalent syntax compares equal
        render_dialect = context.dialect or context.agent_dialect
        exp_components = _ComponentExtractor(parsed_exp, render_dialect).extract()
        gen_components = _ComponentExtractor(parsed_gen, render_dialect).extract()

        clause_scores: dict[str, float] = {}
        differences: dict[str, Any] = {}
        for clause in CLAUSES:
            expected, generated = exp_components[clause], gen_components[clause]
            if not expected and not generated:
                continue
            clause_scores[clause] = round(_f1(expected, generated), 4)
            if clause_scores[clause] < 1.0:
                differences[clause] = {
                    "missing": list((expected - generated).elements())[:_MAX_DIFF_ITEMS],
                    "extra": list((generated - expected).elements())[:_MAX_DIFF_ITEMS],
                }

        score = (
            round(sum(clause_scores.values()) / len(clause_scores), 4)
            if clause_scores
            else 1.0
        )
        return EvaluationResult(
            evaluator_name=self.name,
            score=score,
            passed=score >= 1.0,
            details={
                "composite_score": score,
                "clause_scores": clause_scores,
                "differences": differences,
            },
        )

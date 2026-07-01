"""
component_match_evaluator.py — Evaluates SQL clause similarity using AST parsing via sqlglot.
"""

from __future__ import annotations

import structlog
import sqlglot
import sqlglot.expressions as exp

from src.domain.entities.evaluation_context import EvaluationContext
from src.domain.entities.evaluation_result import EvaluationResult
from src.domain.evaluators.base_evaluator import BaseEvaluator

logger = structlog.get_logger(__name__)


class ComponentMatchEvaluator(BaseEvaluator):
    """
    Compares expected vs generated SQL at clause level: SELECT, FROM, WHERE, GROUP BY, ORDER BY, JOIN.
    """

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

        # Parse ASTs
        parsed_exp = self._safe_parse(context.expected_sql)
        parsed_gen = self._safe_parse(context.generated_sql)

        if not parsed_exp or not parsed_gen:
            return EvaluationResult(
                evaluator_name=self.name,
                score=0.0,
                passed=False,
                details={"reason": "sql_parsing_failed"},
            )

        # Find the Select nodes
        select_exp = parsed_exp.find(exp.Select)
        select_gen = parsed_gen.find(exp.Select)

        # Calculate scores for each clause
        clauses = ["select", "from", "where", "group", "order", "join"]
        scores: dict[str, float] = {}

        for clause in clauses:
            scores[clause] = self._compare_clause(clause, select_exp, select_gen)

        # Average the scores
        avg_score = sum(scores.values()) / len(clauses)
        passed = avg_score >= 1.0  # Pass if perfect match across all clauses

        return EvaluationResult(
            evaluator_name=self.name,
            score=round(avg_score, 4),
            passed=passed,
            details={
                "clause_scores": scores,
                "composite_score": round(avg_score, 4),
            },
        )

    def _safe_parse(self, sql: str) -> exp.Expression | None:
        try:
            return sqlglot.parse_one(sql, read="trino")
        except Exception:
            try:
                return sqlglot.parse_one(sql)
            except Exception:
                return None

    def _compare_clause(self, clause_name: str, exp_select: exp.Select | None, gen_select: exp.Select | None) -> float:
        if not exp_select and not gen_select:
            return 1.0
        if not exp_select or not gen_select:
            return 0.0

        if clause_name == "select":
            exp_node = exp_select.args.get("select")
            gen_node = gen_select.args.get("select")
        elif clause_name == "from":
            exp_node = exp_select.args.get("from")
            gen_node = gen_select.args.get("from")
        elif clause_name == "where":
            exp_node = exp_select.args.get("where")
            gen_node = gen_select.args.get("where")
        elif clause_name == "group":
            exp_node = exp_select.args.get("group")
            gen_node = gen_select.args.get("group")
        elif clause_name == "order":
            exp_node = exp_select.args.get("order")
            gen_node = gen_select.args.get("order")
        elif clause_name == "join":
            # Joins can be a list of Join nodes
            exp_node = exp_select.args.get("joins")
            gen_node = gen_select.args.get("joins")
        else:
            return 0.0

        if not exp_node and not gen_node:
            return 1.0
        if not exp_node or not gen_node:
            return 0.0

        # Special handling for lists of joins
        if clause_name == "join":
            exp_list = exp_node if isinstance(exp_node, list) else [exp_node]
            gen_list = gen_node if isinstance(gen_node, list) else [gen_node]
            exp_sqls = {j.sql(pretty=False).lower().strip() for j in exp_list if j}
            gen_sqls = {j.sql(pretty=False).lower().strip() for j in gen_list if j}
            return 1.0 if exp_sqls == gen_sqls else 0.0

        # For other clauses, check their SQL representations
        def get_sql(node) -> str:
            if isinstance(node, list):
                return ", ".join(n.sql(pretty=False).lower().strip() for n in node if n)
            return node.sql(pretty=False).lower().strip()

        return 1.0 if get_sql(exp_node) == get_sql(gen_node) else 0.0

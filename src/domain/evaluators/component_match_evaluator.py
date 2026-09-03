"""
component_match_evaluator.py — Evaluates SQL clause similarity using AST parsing via sqlglot.

Improvements over v1:
  - Added HAVING and LIMIT clauses.
  - Handles queries with multiple SELECT nodes (subqueries, CTEs, UNIONs) by
    collecting ALL Select nodes via find_all() and averaging scores across them.
  - List-type clauses (SELECT columns, GROUP BY, ORDER BY) are compared as sets
    so that column order differences don't cause false failures.
  - JOIN clauses continue to use set-based comparison (order-independent).
"""

from __future__ import annotations

import structlog
import sqlglot
import sqlglot.expressions as exp

from src.domain.entities.evaluation_context import EvaluationContext
from src.domain.entities.evaluation_result import EvaluationResult
from src.domain.evaluators.base_evaluator import BaseEvaluator

logger = structlog.get_logger(__name__)

# All SQL clauses evaluated per SELECT node.
# Each maps to the sqlglot Select.args key used to retrieve the AST node.
CLAUSE_ARG_KEYS: dict[str, str] = {
    "select":  "expressions",   # list of selected expressions
    "from":    "from",          # FROM clause
    "join":    "joins",         # list of JOIN nodes
    "where":   "where",         # WHERE predicate
    "group":   "group",         # GROUP BY
    "having":  "having",        # HAVING predicate
    "order":   "order",         # ORDER BY
    "limit":   "limit",         # LIMIT / FETCH
}

# Clauses whose nodes form unordered sets (column order should not matter).
SET_BASED_CLAUSES: frozenset[str] = frozenset({"select", "group", "join"})


class ComponentMatchEvaluator(BaseEvaluator):
    """
    Compares expected vs generated SQL at clause level using sqlglot AST parsing.

    Supports multi-SELECT queries (subqueries, CTEs, UNIONs):
      - Collects all Select nodes from both ASTs.
      - Matches them positionally and averages clause scores.
      - If query A has more Select nodes than query B, unmatched ones count as 0.

    Clause comparison:
      - SET_BASED_CLAUSES: converted to a frozenset of SQL strings — column/join
        order does not affect the score.
      - Other clauses: exact SQL string comparison (preserves ORDER BY sequence).
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

        parsed_exp = self._safe_parse(context.expected_sql)
        parsed_gen = self._safe_parse(context.generated_sql)

        if not parsed_exp or not parsed_gen:
            return EvaluationResult(
                evaluator_name=self.name,
                score=0.0,
                passed=False,
                details={"reason": "sql_parsing_failed"},
            )

        # Collect ALL Select nodes (handles subqueries, CTEs, UNIONs)
        exp_selects: list[exp.Select] = list(parsed_exp.find_all(exp.Select))
        gen_selects: list[exp.Select] = list(parsed_gen.find_all(exp.Select))

        if not exp_selects and not gen_selects:
            return EvaluationResult(
                evaluator_name=self.name,
                score=1.0,
                passed=True,
                details={"reason": "no_select_nodes_in_either"},
            )

        # Match SELECT nodes positionally; pad with None for unmatched ones
        max_len = max(len(exp_selects), len(gen_selects))
        exp_selects += [None] * (max_len - len(exp_selects))
        gen_selects += [None] * (max_len - len(gen_selects))

        all_clause_scores: list[dict[str, float]] = []

        for exp_select, gen_select in zip(exp_selects, gen_selects):
            clause_scores: dict[str, float] = {}
            for clause_name, arg_key in CLAUSE_ARG_KEYS.items():
                clause_scores[clause_name] = self._compare_clause(
                    clause_name, arg_key, exp_select, gen_select
                )
            all_clause_scores.append(clause_scores)

        # Aggregate: average across all SELECT nodes, then across all clauses
        n_clauses = len(CLAUSE_ARG_KEYS)
        total_score = sum(
            sum(cs.values()) / n_clauses
            for cs in all_clause_scores
        ) / max_len

        avg_score = round(total_score, 4)
        passed = avg_score >= 1.0

        details: dict = {
            "composite_score": avg_score,
            "select_node_count": max_len,
        }
        if max_len == 1:
            # Single SELECT: surface per-clause scores directly for readability
            details["clause_scores"] = all_clause_scores[0]
        else:
            details["per_select_clause_scores"] = all_clause_scores

        return EvaluationResult(
            evaluator_name=self.name,
            score=avg_score,
            passed=passed,
            details=details,
        )

    def _safe_parse(self, sql: str) -> exp.Expression | None:
        try:
            return sqlglot.parse_one(sql, read="trino")
        except Exception:
            try:
                return sqlglot.parse_one(sql)
            except Exception:
                return None

    def _compare_clause(
        self,
        clause_name: str,
        arg_key: str,
        exp_select: exp.Select | None,
        gen_select: exp.Select | None,
    ) -> float:
        # Both missing → perfect match (neither query uses this clause)
        if not exp_select and not gen_select:
            return 1.0
        # One missing → mismatch
        if not exp_select or not gen_select:
            return 0.0

        exp_node = exp_select.args.get(arg_key)
        gen_node = gen_select.args.get(arg_key)

        # Both clause nodes absent → both queries omit this clause
        if not exp_node and not gen_node:
            return 1.0
        # Only one has the clause → structural mismatch
        if not exp_node or not gen_node:
            return 0.0

        # Normalise to list
        exp_list = exp_node if isinstance(exp_node, list) else [exp_node]
        gen_list = gen_node if isinstance(gen_node, list) else [gen_node]

        exp_sqls = [n.sql(pretty=False).lower().strip() for n in exp_list if n]
        gen_sqls = [n.sql(pretty=False).lower().strip() for n in gen_list if n]

        if clause_name in SET_BASED_CLAUSES:
            # Order-independent comparison (SELECT columns, GROUP BY, JOINs)
            return 1.0 if set(exp_sqls) == set(gen_sqls) else 0.0

        # Ordered comparison (WHERE, HAVING, ORDER BY, LIMIT, FROM)
        return 1.0 if exp_sqls == gen_sqls else 0.0

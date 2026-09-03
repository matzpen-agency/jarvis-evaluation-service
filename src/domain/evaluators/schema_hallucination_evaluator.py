"""
schema_hallucination_evaluator.py — Evaluates whether the generated SQL references non-existent tables/columns.
"""

from __future__ import annotations

import structlog
import sqlglot
import sqlglot.expressions as exp
from typing import Any

from src.domain.entities.evaluation_context import EvaluationContext
from src.domain.entities.evaluation_result import EvaluationResult
from src.domain.evaluators.base_evaluator import BaseEvaluator

logger = structlog.get_logger(__name__)


class SchemaHallucinationEvaluator(BaseEvaluator):
    """
    Detects table and column hallucinations by comparing referenced objects in the generated SQL
    against the allowed schema and column metadata fetched from the backend.
    """

    def __init__(self, table_resolver: Any) -> None:
        self._resolver = table_resolver

    @property
    def name(self) -> str:
        return "schema_hallucination"

    async def evaluate(self, context: EvaluationContext) -> EvaluationResult:
        try:
            return await self._evaluate(context)
        except Exception as exc:
            logger.error("schema_hallucination_evaluator.error", error=str(exc))
            return EvaluationResult.from_error(self.name, str(exc))

    async def _evaluate(self, context: EvaluationContext) -> EvaluationResult:
        if not context.generated_sql:
            return EvaluationResult(
                evaluator_name=self.name,
                score=0.0,
                passed=False,
                details={"reason": "no_generated_sql"},
            )

        # 1. Fetch the catalog/schema map from the backend resolver
        column_check_available = True
        try:
            schema_map = await self._resolver.get_table_schema_map()
            if not schema_map:
                column_check_available = False
        except Exception as exc:
            logger.warning("schema_hallucination.resolver_failed", error=str(exc))
            schema_map = {}
            column_check_available = False

        # 2. Extract referenced tables and columns from the generated query
        referenced = self._extract_tables_and_columns(context.generated_sql)

        hallucinated_tables: list[str] = []
        hallucinated_columns: list[str] = []

        allowed_lower = {t.lower().strip() for t in context.allowed_tables}

        # Check references
        for tab, cols in referenced.items():
            if not tab:
                # Unassociated columns: check if they exist in any of the allowed tables
                for col in cols:
                    exists = False
                    for allowed in allowed_lower:
                        if col in schema_map.get(allowed, {}):
                            exists = True
                            break
                    if not exists and allowed_lower:
                        # Only report column hallucination if we resolved columns for at least one table
                        has_metadata = any(len(schema_map.get(allowed, {})) > 0 for allowed in allowed_lower)
                        if has_metadata:
                            hallucinated_columns.append(f"unassociated.{col}")
                continue

            # Verify table is allowed (case-insensitive, matching simple or qualified name)
            is_allowed = False
            matched_allowed_tab = None
            for allowed in allowed_lower:
                if tab == allowed or tab.endswith(f".{allowed}"):
                    is_allowed = True
                    matched_allowed_tab = allowed
                    break

            if not is_allowed:
                hallucinated_tables.append(tab)
            else:
                # Table is allowed, check columns if schema metadata is available
                valid_cols = schema_map.get(matched_allowed_tab, {})
                if not valid_cols:
                    # Try qualified key in schema map
                    valid_cols = schema_map.get(tab, {})

                if valid_cols:
                    for col in cols:
                        if col not in valid_cols:
                            hallucinated_columns.append(f"{tab}.{col}")

        score = 1.0 if not hallucinated_tables and not hallucinated_columns else 0.0
        passed = score == 1.0

        return EvaluationResult(
            evaluator_name=self.name,
            score=score,
            passed=passed,
            details={
                "hallucinated_tables": hallucinated_tables,
                "hallucinated_columns": hallucinated_columns,
                "referenced_tables_columns": {t: list(c) for t, c in referenced.items()},
                "column_check_available": column_check_available,
            },
        )

    def _extract_tables_and_columns(self, sql: str) -> dict[str, set[str]]:
        try:
            parsed = sqlglot.parse_one(sql, read="trino")
        except Exception:
            try:
                parsed = sqlglot.parse_one(sql)
            except Exception:
                return {}

        referenced: dict[str, set[str]] = {}
        tables = [t.name.lower().strip() for t in parsed.find_all(exp.Table)]
        for t in tables:
            referenced[t] = set()

        for c in parsed.find_all(exp.Column):
            col_name = c.name.lower().strip()
            tab_name = c.text("table").lower().strip()

            if tab_name:
                if tab_name in referenced:
                    referenced[tab_name].add(col_name)
                else:
                    # Handle schema name matching or nested tables
                    referenced[tab_name] = {col_name}
            else:
                if len(tables) == 1:
                    referenced[tables[0]].add(col_name)
                else:
                    if "" not in referenced:
                        referenced[""] = set()
                    referenced[""].add(col_name)

        return referenced

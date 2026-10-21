"""
schema_hallucination_evaluator.py — Detects tables/columns in the generated SQL
that don't exist in the connection's schema.

The schema comes from the connection metadata (context.schema_map, falling
back to the table resolver). References are resolved per query scope with
sqlglot, so table aliases, CTE names, subquery columns and SELECT aliases
used in ORDER BY are not mistaken for hallucinations.

Score: 1.0 when every referenced table and column exists, 0.0 otherwise.
Not applicable when no schema metadata is available or there is no SQL.
"""

from __future__ import annotations

from typing import Any

import sqlglot.expressions as exp
import structlog
from sqlglot.optimizer.scope import traverse_scope

from src.domain.entities.evaluation_context import EvaluationContext
from src.domain.entities.evaluation_result import EvaluationResult
from src.domain.evaluators.base_evaluator import BaseEvaluator
from src.domain.evaluators.sql_comparison_utils import safe_parse

logger = structlog.get_logger(__name__)

_DERIVED = object()  # a source whose columns we can't check (CTE, subquery, UNNEST)


class SchemaHallucinationEvaluator(BaseEvaluator):
    """Flags generated SQL that references non-existent tables or columns."""

    def __init__(self, table_resolver: Any = None) -> None:
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
            return EvaluationResult.not_applicable(self.name, "no_generated_sql")

        schema = self._lower_schema(await self._schema_map(context))
        if not schema:
            return EvaluationResult.not_applicable(self.name, "schema_metadata_unavailable")

        parsed = safe_parse(context.generated_sql, context.agent_dialect, context.dialect)
        if parsed is None:
            return EvaluationResult.not_applicable(self.name, "sql_parsing_failed")

        hallucinated_tables: set[str] = set()
        hallucinated_columns: set[str] = set()
        referenced: dict[str, set[str]] = {}

        for scope in traverse_scope(parsed):
            sources: dict[str, Any] = {}
            for alias, (_, source) in scope.selected_sources.items():
                if isinstance(source, exp.Table):
                    columns = self._lookup(source, schema)
                    if columns is None:
                        hallucinated_tables.add(self._display_name(source))
                    else:
                        referenced.setdefault(source.name.lower(), set())
                    sources[alias.lower()] = (source.name.lower(), columns)
                else:
                    sources[alias.lower()] = (alias.lower(), _DERIVED)

            select_aliases = set()
            if isinstance(scope.expression, exp.Select):
                select_aliases = {
                    p.alias.lower() for p in scope.expression.expressions if p.alias
                }

            for column in scope.columns:
                if isinstance(column.this, exp.Star):
                    continue
                name = column.name.lower()
                qualifier = column.table.lower()
                if qualifier:
                    if qualifier not in sources:
                        continue  # correlated reference to an outer scope
                    table_name, columns = sources[qualifier]
                    if columns is None or columns is _DERIVED:
                        continue
                    referenced[table_name].add(name)
                    if name not in columns:
                        hallucinated_columns.add(f"{table_name}.{name}")
                    continue

                if name in select_aliases and not any(
                    isinstance(cols, set) and name in cols for _, cols in sources.values()
                ):
                    continue  # e.g. ORDER BY revenue → SUM(...) AS revenue
                known = [(t, cols) for t, cols in sources.values() if isinstance(cols, set)]
                if not known or any(cols is _DERIVED for _, cols in sources.values()):
                    continue
                owners = [t for t, cols in known if name in cols]
                if owners:
                    referenced[owners[0]].add(name)
                elif len(known) == len(sources):
                    hallucinated_columns.add(name)

        passed = not hallucinated_tables and not hallucinated_columns
        return EvaluationResult(
            evaluator_name=self.name,
            score=1.0 if passed else 0.0,
            passed=passed,
            details={
                "hallucinated_tables": sorted(hallucinated_tables),
                "hallucinated_columns": sorted(hallucinated_columns),
                "referenced_tables_columns": {t: sorted(c) for t, c in referenced.items()},
            },
        )

    async def _schema_map(self, context: EvaluationContext) -> dict:
        if context.schema_map:
            return context.schema_map
        if self._resolver is None:
            return {}
        try:
            return await self._resolver.get_table_schema_map(context.connection_id)
        except Exception as exc:
            logger.warning("schema_hallucination.resolver_failed", error=str(exc))
            return {}

    @staticmethod
    def _lower_schema(schema_map: dict) -> dict[str, set[str]]:
        return {
            str(table).lower(): {str(c).lower() for c in columns}
            for table, columns in (schema_map or {}).items()
        }

    @staticmethod
    def _lookup(table: exp.Table, schema: dict[str, set[str]]) -> set[str] | None:
        """
        Columns of the referenced table, or None if it doesn't exist.

        A schema-qualified reference must match the schema; the catalog is
        ignored because it names the engine-side catalog (e.g. Trino's
        "postgres"), not part of the connection's own schema.
        """
        name = table.name.lower()
        db = (table.db or "").lower()
        if db:
            qualified = schema.get(f"{db}.{name}")
            if qualified is not None:
                return qualified
            has_schema_keys = any(k.endswith(f".{name}") for k in schema)
            return None if has_schema_keys else schema.get(name)
        return schema.get(name)

    @staticmethod
    def _display_name(table: exp.Table) -> str:
        parts = [table.catalog, table.db, table.name]
        return ".".join(p for p in parts if p).lower()

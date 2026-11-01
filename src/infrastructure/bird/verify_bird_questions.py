"""
verify_bird_questions.py — Check which BIRD questions keep their answer.

BIRD gold SQL is written for SQLite. After load_bird_databases.py copies a
database into Postgres or Trino, some gold SQL can't run there or returns
something else (SQLite is lax about types, GROUP BY…). This runs every gold
SQL of the bundled databases the way the evaluation runs it (translated by
adapt_sqlite_benchmark_sql, then to Trino when the target is Trino), compares
the result with SQLite's, and with --write saves the questions that match,
per engine, to src/config/bird/databases/verified_questions.json. The BIRD
dataset only uses questions that match on every engine saved there, so a gold
SQL that breaks in translation never counts against the agent.

    python -m src.infrastructure.bird.verify_bird_questions --catalog minio [--write]
    python -m src.infrastructure.bird.verify_bird_questions --target postgres --dsn ...

For databases pushed to dataverse with push_bird_to_oasis.py, point it at the
SQLite files and the pool's schema; the matches are saved as engine
"dataverse", which the BIRD dataset uses when BIRD_DATAVERSE_SCHEMA is set:

    python -m src.infrastructure.bird.verify_bird_questions --catalog dataverse \
        --dataverse-schema agency__bird-public-benchmark --databases-dir /data/bird --write
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

from src.domain.entities.query_result import QueryResult
from src.domain.evaluators.sql_comparison_utils import (
    adapt_sqlite_benchmark_sql,
    clean_name,
    compare_results,
    parse_grading_method,
    transpile_sql,
)
from src.infrastructure.bird.bird_dataset_loader import (
    DATAVERSE_ENGINE,
    VERIFIED_QUESTIONS_FILE,
)
from src.infrastructure.bird.load_bird_databases import (
    DATABASES_DIR,
    DEFAULT_SCHEMA_PREFIX,
    _say,
    _trino_cursor,
    available_databases,
    schema_name,
)

QUESTIONS_PATH = DATABASES_DIR.parent / "bird_verified_cot_2462.jsonl"
VERIFIED_PATH = DATABASES_DIR / VERIFIED_QUESTIONS_FILE


def gold_sql_for(
    sql: str, schema: str, target: str, db_id: str | None = None
) -> str | None:
    """
    The gold SQL as the evaluation runs it on a Postgres connection / the
    agent's Trino; with db_id, on the database pushed to dataverse in `schema`.
    """
    naming = (
        {"table_prefix": f"{clean_name(db_id)}__", "clean_names": True} if db_id else {}
    )
    adapted = adapt_sqlite_benchmark_sql(sql, schema, "postgres", **naming)
    if adapted is None or target == "postgres":
        return adapted
    return transpile_sql(adapted, read="postgres", write="trino")


def _result(rows: list) -> QueryResult:
    width = len(rows[0]) if rows else 0
    return QueryResult(
        success=True,
        rows=[list(r) for r in rows],
        columns=[f"c{i}" for i in range(width)],
    )


def verify(
    run_query,
    target: str,
    databases: list[str],
    prefix: str,
    databases_dir: Path = DATABASES_DIR,
    dataverse_schema: str = "",
) -> dict[str, list[int]]:
    questions = [json.loads(line) for line in QUESTIONS_PATH.open(encoding="utf-8")]
    verified: dict[str, list[int]] = {}
    for db_id in databases:
        lite = sqlite3.connect(
            f"file:{databases_dir / (db_id + '.sqlite')}?mode=ro", uri=True
        )
        schema = dataverse_schema or schema_name(db_id, prefix)
        ok, failures = [], []
        for q in (q for q in questions if q["db_id"] == db_id):
            expected = _result(lite.execute(q["SQL"]).fetchall())
            sql = gold_sql_for(
                q["SQL"], schema, target, db_id if dataverse_schema else None
            )
            try:
                actual = _result(run_query(sql)) if sql else None
            except Exception as exc:
                failures.append((q["question_id"], str(exc).splitlines()[0][:100]))
                continue
            grading = parse_grading_method(q.get("grading_method"))
            score, _ = compare_results(
                expected,
                actual,
                allow_extra_columns=False,
                requires_ordering=grading.mode == "list",
                grading=grading,
            )
            if score >= 1.0:
                ok.append(q["question_id"])
            else:
                failures.append((q["question_id"], "different result"))
        lite.close()
        verified[db_id] = ok
        _say(
            f"{db_id}: {len(ok)}/{len(ok) + len(failures)} questions keep their answer"
        )
        for question_id, reason in failures:
            _say(f"    {question_id}: {reason}")
    return verified


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("databases", nargs="*")
    parser.add_argument("--target", choices=("trino", "postgres"), default="trino")
    parser.add_argument("--catalog", help="Trino catalog holding the BIRD schemas")
    parser.add_argument("--dsn", help="Postgres connection string")
    parser.add_argument("--schema-prefix", default=DEFAULT_SCHEMA_PREFIX)
    parser.add_argument("--databases-dir", type=Path, default=DATABASES_DIR)
    parser.add_argument(
        "--dataverse-schema",
        default="",
        help="the databases were pushed to dataverse in this schema (push_bird_to_oasis)",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help=f"save the matching questions to {VERIFIED_PATH}",
    )
    args = parser.parse_args(argv)
    databases = args.databases or available_databases(args.databases_dir)
    if args.dataverse_schema and args.target != "trino":
        parser.error("--dataverse-schema needs --target trino")

    if args.target == "trino":
        if not args.catalog:
            parser.error("--catalog is required with --target trino")
        cursor = _trino_cursor(args.catalog)

        def run_query(sql: str) -> list:
            cursor.execute(sql)
            return cursor.fetchall()
    else:
        if not args.dsn:
            parser.error("--dsn is required with --target postgres")
        import psycopg2

        pg = psycopg2.connect(args.dsn)
        pg.autocommit = True

        def run_query(sql: str) -> list:
            with pg.cursor() as cur:
                cur.execute(sql)
                return cur.fetchall()

    verified = verify(
        run_query,
        args.target,
        databases,
        args.schema_prefix,
        args.databases_dir,
        args.dataverse_schema,
    )
    if args.write:
        existing = (
            json.loads(VERIFIED_PATH.read_text()) if VERIFIED_PATH.exists() else {}
        )
        engine = DATAVERSE_ENGINE if args.dataverse_schema else args.target
        existing.setdefault(engine, {}).update(verified)
        VERIFIED_PATH.write_text(json.dumps(existing, indent=2, sort_keys=True) + "\n")
        _say(f"Saved to {Path(VERIFIED_PATH)}")
    if not any(verified.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()

"""
convert_bird_trino_questions.py — Gold SQL of the t2s_bird_trino_he questions
for the tables the BIRD databases were loaded into.

The bundle's `trino_sql` reads each database as its own schema <db_id> with
the original lower-case names ("county name"), and names no schema. This adds
`target_sql` to every question, for one of the two layouts:

  dataverse  the databases pushed with push_bird_to_oasis.py:
               "agency__bird-public-benchmark"."<db-id>-<table>", cleaned columns
  postgres   the databases loaded by load_bird_databases.py --schema-prefix "":
               the same SQL, its tables put in schema <db_id> (so one Jeen
               connection can hold every database)

    python -m src.infrastructure.bird.convert_bird_trino_questions \\
        bundle/splits/dev.jsonl bundle/splits/train.jsonl --out bird_questions.jsonl

--verify also runs every target_sql through the agent's Trino (AGENT_TRINO_*,
--catalog) and compares it with the original SQLite gold on the SQLite file
(--databases-dir); each question gets "check": "ok", "different result" or the
error, and --only-ok keeps the questions that passed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.domain.entities.query_result import QueryResult
from src.domain.evaluators.sql_comparison_utils import (
    compare_results,
    qualify_tables,
    to_dataverse_sql,
)
from src.infrastructure.bird.load_bird_databases import (
    _say,
    _trino_cursor,
    find_databases,
    open_database,
)
from src.infrastructure.bird.push_bird_to_oasis import TRINO_SCHEMA


def target_sql(row: dict, layout: str, schema: str = TRINO_SCHEMA) -> str | None:
    db_id = row["db_id"].lower()
    if layout == "dataverse":
        return to_dataverse_sql(row["trino_sql"], schema, db_id)
    return qualify_tables(row["trino_sql"], db_id, "trino")


def _result(rows: list) -> QueryResult:
    width = len(rows[0]) if rows else 0
    return QueryResult(
        success=True,
        rows=[list(r) for r in rows],
        columns=[f"c{i}" for i in range(width)],
    )


def check(row: dict, run_query, sqlite_dir: Path, connections: dict) -> str:
    """Compare target_sql's result with the SQLite gold's, as rows sets."""
    db_id = row["db_id"]
    if db_id not in connections:
        path = find_databases(sqlite_dir).get(db_id)
        connections[db_id] = open_database(path) if path else None
    lite = connections[db_id]
    if lite is None:
        return f"no SQLite file for {db_id}"
    if not row.get("target_sql"):
        return "can't parse trino_sql"
    expected = _result(lite.execute(row["sqlite_sql"]).fetchall())
    try:
        actual = _result(run_query(row["target_sql"]))
    except Exception as exc:
        return str(exc).splitlines()[0][:200]
    score, _ = compare_results(expected, actual, allow_extra_columns=False)
    return "ok" if score >= 1.0 else "different result"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "questions", nargs="+", type=Path, help="splits/<split>.jsonl files"
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--layout", choices=("dataverse", "postgres"), default="dataverse"
    )
    parser.add_argument(
        "--schema", default=TRINO_SCHEMA, help="dataverse schema of the pool"
    )
    parser.add_argument("--databases", nargs="*", help="only these db_ids")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--catalog", help="--verify: Trino catalog, e.g. dataverse")
    parser.add_argument("--databases-dir", type=Path, help="--verify: the SQLite files")
    parser.add_argument(
        "--only-ok", action="store_true", help="keep questions that passed"
    )
    args = parser.parse_args(argv)
    if args.verify and not (args.catalog and args.databases_dir):
        parser.error("--verify needs --catalog and --databases-dir")

    rows = [
        json.loads(line)
        for path in args.questions
        for line in path.open(encoding="utf-8")
    ]
    if args.databases:
        rows = [r for r in rows if r["db_id"] in set(args.databases)]

    run_query = None
    if args.verify:
        cursor = _trino_cursor(args.catalog)

        def run_query(sql: str) -> list:
            cursor.execute(sql)
            return cursor.fetchall()

    counts: dict[str, int] = {}
    connections: dict = {}
    with args.out.open("w", encoding="utf-8") as out:
        for n, row in enumerate(rows, 1):
            row["target_sql"] = target_sql(row, args.layout, args.schema)
            if run_query:
                row["check"] = check(row, run_query, args.databases_dir, connections)
                status = (
                    row["check"]
                    if row["check"] in ("ok", "different result")
                    else "error"
                )
                counts[status] = counts.get(status, 0) + 1
                if row["check"] != "ok":
                    _say(f"  {row['db_id']} #{row.get('index')}: {row['check']}")
                if n % 100 == 0:
                    _say(f"{n}/{len(rows)} checked: {counts}")
                if args.only_ok and row["check"] != "ok":
                    continue
            elif row["target_sql"] is None:
                counts["unparsed"] = counts.get("unparsed", 0) + 1
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
    _say(f"{len(rows)} questions → {args.out} {counts or ''}")
    if run_query and counts.get("ok", 0) == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()

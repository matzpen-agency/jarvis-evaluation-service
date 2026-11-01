"""
load_bird_databases.py — Copy the bundled BIRD databases into a database.

BIRD questions can only be evaluated on a data connection holding their
database. The service ships a few BIRD databases as SQLite files
(src/config/bird/databases/<db_id>.sqlite, see ATTRIBUTION.md) so they can be
loaded inside a private network with no internet access. Each database becomes
one schema, "bird_<db_id>" (BIRD_SCHEMA_PREFIX), with lower-case table and
column names. Dates stay ISO text, as SQLite stores them: BIRD's gold SQL
compares them with string literals, which Trino refuses for DATE columns.

Two targets:
  trino     (default) through the agent's Trino (AGENT_TRINO_* settings), into
            a catalog the Trino user can write to:
              python -m src.infrastructure.bird.load_bird_databases --catalog dataverse
  postgres  straight into Postgres; also adds primary and foreign keys, which
            Jeen uses to find joins:
              python -m src.infrastructure.bird.load_bird_databases \\
                  --target postgres --dsn postgresql://user:pass@host:5432/db

Re-running replaces the tables. Afterwards add (or refresh) the Jeen connection
that reads these schemas and run the dataset "bird" on it.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DATABASES_DIR = Path(__file__).resolve().parents[2] / "config" / "bird" / "databases"
DEFAULT_SCHEMA_PREFIX = "bird_"

# Logical column types → (Trino type, Postgres type)
_TYPES = {
    "bigint": ("BIGINT", "BIGINT"),
    "double": ("DOUBLE", "DOUBLE PRECISION"),
    "binary": ("VARBINARY", "BYTEA"),
    "text": ("VARCHAR", "TEXT"),
}


@dataclass
class Column:
    name: str
    kind: str  # key of _TYPES


@dataclass
class Table:
    name: str
    columns: list[Column]
    primary_key: list[str] = field(default_factory=list)
    # (columns, referenced table, referenced columns)
    foreign_keys: list[tuple[list[str], str, list[str]]] = field(default_factory=list)


def available_databases(directory: Path = DATABASES_DIR) -> list[str]:
    return sorted(p.stem for p in directory.glob("*.sqlite"))


def schema_name(db_id: str, prefix: str = DEFAULT_SCHEMA_PREFIX) -> str:
    return f"{prefix}{db_id}".lower()


def _declared_kind(declared: str) -> str:
    d = (declared or "").upper()
    if "INT" in d:
        return "bigint"
    if any(t in d for t in ("REAL", "FLOA", "DOUB", "NUMERIC", "DECIMAL")):
        return "double"
    if "BLOB" in d:
        return "binary"
    return "text"


def _fits(kind: str, value: Any) -> bool:
    if kind == "bigint":
        return isinstance(value, int) and not isinstance(value, bool)
    if kind == "double":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if kind == "binary":
        return isinstance(value, bytes)
    return True


def read_tables(con: sqlite3.Connection) -> list[Table]:
    """Tables with lower-case names and the narrowest type every value fits (else text)."""
    names = [
        r[0]
        for r in con.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%' "
            "ORDER BY name"
        )
    ]
    tables = []
    for name in names:
        info = con.execute(f'PRAGMA table_info("{name}")').fetchall()
        columns = []
        for _, col, declared, _, _, _ in info:
            kind = _declared_kind(declared)
            if kind != "text":
                values = con.execute(
                    f'SELECT "{col}" FROM "{name}" WHERE "{col}" IS NOT NULL'
                )
                if not all(_fits(kind, v) for (v,) in values):
                    kind = "text"
            columns.append(Column(col.lower(), kind))
        pk = [row[1].lower() for row in sorted(info, key=lambda r: r[5]) if row[5]]
        fks: dict[int, tuple[list[str], str, list[str]]] = {}
        for fk_id, _, ref_table, src, ref_col, *_ in con.execute(
            f'PRAGMA foreign_key_list("{name}")'
        ):
            if ref_col is None:  # references the other table's primary key
                continue
            cols, _, ref_cols = fks.setdefault(fk_id, ([], ref_table.lower(), []))
            cols.append(src.lower())
            ref_cols.append(ref_col.lower())
        tables.append(Table(name.lower(), columns, pk, list(fks.values())))
    return tables


def read_rows(con: sqlite3.Connection, original_name: str) -> Iterator[tuple]:
    yield from con.execute(f'SELECT * FROM "{original_name}"')


def convert(value: Any, kind: str) -> Any:
    if value is None:
        return None
    if kind == "text" and not isinstance(value, str):
        return (
            value.decode("utf-8", "replace") if isinstance(value, bytes) else str(value)
        )
    if kind == "double":
        return float(value)
    return value


def _say(message: str) -> None:
    sys.stdout.write(message + "\n")
    sys.stdout.flush()


def _ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


# ── Trino ─────────────────────────────────────────────────────────────────────


def _trino_literal(value: Any, kind: str) -> str:
    if value is None:
        return "NULL"
    if kind == "bigint":
        return str(value)
    if kind == "double":
        return f"DOUBLE '{value!r}'"
    if kind == "binary":
        return f"X'{value.hex()}'"
    return "'" + value.replace("'", "''") + "'"


def load_into_trino(
    cursor,
    catalog: str,
    schema: str,
    con: sqlite3.Connection,
    batch_size: int = 500,
    log=None,
) -> None:
    log = log or _say
    run = lambda sql: (cursor.execute(sql), cursor.fetchall())  # noqa: E731
    target = f"{_ident(catalog)}.{_ident(schema)}"
    run(f"CREATE SCHEMA IF NOT EXISTS {target}")
    tables = read_tables(con)
    originals = _original_names(con)
    for table in tables:
        full = f"{target}.{_ident(table.name)}"
        run(f"DROP TABLE IF EXISTS {full}")
        cols = ", ".join(f"{_ident(c.name)} {_TYPES[c.kind][0]}" for c in table.columns)
        run(f"CREATE TABLE {full} ({cols})")
        count = 0
        for batch in _batches(read_rows(con, originals[table.name]), batch_size):
            values = ", ".join(
                "("
                + ", ".join(
                    _trino_literal(convert(v, c.kind), c.kind)
                    for v, c in zip(row, table.columns, strict=True)
                )
                + ")"
                for row in batch
            )
            run(f"INSERT INTO {full} VALUES {values}")
            count += len(batch)
        log(f"  {schema}.{table.name}: {count} rows")
    # Keys can't be declared through Trino; a PostgreSQL catalog can still pass
    # them to the database. Best effort: other catalogs simply skip them.
    for statement in _key_statements(schema, tables):
        try:
            run(
                f"CALL {_ident(catalog)}.system.execute(query => '{statement.replace(chr(39), chr(39) * 2)}')"
            )
        except Exception as exc:
            log(f"  keys skipped ({type(exc).__name__}): the catalog can't add them")
            break


# ── Postgres ──────────────────────────────────────────────────────────────────


def load_into_postgres(pg, schema: str, con: sqlite3.Connection, log=None) -> None:
    log = log or _say
    from psycopg2.extras import execute_values

    tables = read_tables(con)
    originals = _original_names(con)
    with pg.cursor() as cur:
        cur.execute(f"DROP SCHEMA IF EXISTS {_ident(schema)} CASCADE")
        cur.execute(f"CREATE SCHEMA {_ident(schema)}")
        for table in tables:
            full = f"{_ident(schema)}.{_ident(table.name)}"
            cols = ", ".join(
                f"{_ident(c.name)} {_TYPES[c.kind][1]}" for c in table.columns
            )
            cur.execute(f"CREATE TABLE {full} ({cols})")
            rows = [
                tuple(
                    convert(v, c.kind) for v, c in zip(row, table.columns, strict=True)
                )
                for row in read_rows(con, originals[table.name])
            ]
            execute_values(cur, f"INSERT INTO {full} VALUES %s", rows, page_size=1000)
            log(f"  {schema}.{table.name}: {len(rows)} rows")
        for statement in _key_statements(schema, tables):
            cur.execute("SAVEPOINT key")
            try:
                cur.execute(statement)
            except Exception as exc:
                cur.execute("ROLLBACK TO SAVEPOINT key")
                log(f"  key skipped: {statement[:90]}… ({str(exc).splitlines()[0]})")
        cur.execute("ANALYZE")
    pg.commit()


def _key_statements(schema: str, tables: list[Table]) -> list[str]:
    """Primary keys, then foreign keys (NOT VALID: BIRD data doesn't always satisfy them)."""
    statements = []
    for t in tables:
        if t.primary_key:
            statements.append(
                f"ALTER TABLE {_ident(schema)}.{_ident(t.name)} "
                f"ADD PRIMARY KEY ({', '.join(map(_ident, t.primary_key))})"
            )
    for t in tables:
        for i, (cols, ref_table, ref_cols) in enumerate(t.foreign_keys):
            statements.append(
                f"ALTER TABLE {_ident(schema)}.{_ident(t.name)} "
                f"ADD CONSTRAINT {_ident(f'fk_{t.name}_{i}')} "
                f"FOREIGN KEY ({', '.join(map(_ident, cols))}) "
                f"REFERENCES {_ident(schema)}.{_ident(ref_table)} "
                f"({', '.join(map(_ident, ref_cols))}) NOT VALID"
            )
    return statements


def _original_names(con: sqlite3.Connection) -> dict[str, str]:
    return {
        r[0].lower(): r[0]
        for r in con.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
    }


def _batches(rows: Iterator[tuple], size: int) -> Iterator[list[tuple]]:
    batch: list[tuple] = []
    for row in rows:
        batch.append(row)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def _trino_cursor(catalog: str):
    import trino

    from src.config.settings import Settings

    s = Settings()
    if not s.AGENT_TRINO_HOST:
        sys.exit(
            "AGENT_TRINO_HOST is not set; use --target postgres or set the AGENT_TRINO_* variables"
        )
    auth = None
    if s.AGENT_TRINO_CERT_PATH and s.AGENT_TRINO_KEY_PATH:
        auth = trino.auth.CertificateAuthentication(
            s.AGENT_TRINO_CERT_PATH, s.AGENT_TRINO_KEY_PATH
        )
    verify = s.AGENT_TRINO_VERIFY
    if isinstance(verify, str) and verify.lower() in ("true", "false"):
        verify = verify.lower() == "true"
    return trino.dbapi.connect(
        host=s.AGENT_TRINO_HOST,
        port=s.AGENT_TRINO_PORT,
        user=s.AGENT_TRINO_USER,
        catalog=catalog,
        http_scheme=s.AGENT_TRINO_HTTP_SCHEME,
        auth=auth,
        verify=verify,
        request_timeout=max(s.AGENT_TRINO_REQUEST_TIMEOUT, 300),
    ).cursor()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "databases",
        nargs="*",
        help=f"BIRD db_ids (default: all bundled: {', '.join(available_databases())})",
    )
    parser.add_argument("--target", choices=("trino", "postgres"), default="trino")
    parser.add_argument(
        "--catalog", help="Trino catalog to write into (target trino), e.g. dataverse"
    )
    parser.add_argument("--dsn", help="Postgres connection string (target postgres)")
    parser.add_argument("--schema-prefix", default=DEFAULT_SCHEMA_PREFIX)
    args = parser.parse_args(argv)

    databases = args.databases or available_databases()
    unknown = sorted(set(databases) - set(available_databases()))
    if unknown:
        parser.error(
            f"not bundled: {', '.join(unknown)} (bundled: {', '.join(available_databases())})"
        )

    if args.target == "trino":
        if not args.catalog:
            parser.error("--catalog is required with --target trino")
        cursor = _trino_cursor(args.catalog)
    else:
        if not args.dsn:
            parser.error("--dsn is required with --target postgres")
        import psycopg2

        pg = psycopg2.connect(args.dsn)

    for db_id in databases:
        schema = schema_name(db_id, args.schema_prefix)
        _say(f"Loading {db_id} into {schema}…")
        con = sqlite3.connect(
            f"file:{DATABASES_DIR / (db_id + '.sqlite')}?mode=ro", uri=True
        )
        try:
            if args.target == "trino":
                load_into_trino(cursor, args.catalog, schema, con)
            else:
                load_into_postgres(pg, schema, con)
        finally:
            con.close()
    _say(
        "Done. Add or refresh the Jeen connection that reads these schemas, then run dataset 'bird' on it."
    )


if __name__ == "__main__":
    main()

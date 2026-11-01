"""
test_bird_bundled_databases.py — BIRD databases shipped with the service:
copying them into Postgres/Trino, running their gold SQL there, and the
"bird" dataset on a connection that holds them.
"""

from __future__ import annotations

import json
import sqlite3

import pytest

from src.domain.entities.dataset_item import DatasetItem
from src.domain.entities.evaluation_context import EvaluationContext
from src.domain.evaluators.sql_comparison_utils import (
    adapt_sqlite_benchmark_sql,
    qualify_tables,
    transpile_sql,
)
from src.domain.services.evaluation_runner import EvaluationRunner
from src.infrastructure.bird.bird_benchmark_provider import BirdBenchmarkProvider
from src.infrastructure.bird.bird_dataset_loader import BirdDatasetLoader
from src.infrastructure.bird.load_bird_databases import (
    _key_statements,
    _trino_literal,
    read_tables,
)
from src.ports.dataset_provider import DatasetNotFoundError

ROWS = [
    {
        "question_id": 1,
        "db_id": "shop",
        "question": "q1",
        "SQL": "SELECT 1",
        "grading_method": "set",
    },
    {
        "question_id": 2,
        "db_id": "shop",
        "question": "q2",
        "SQL": "SELECT 2",
        "grading_method": "set",
    },
    {
        "question_id": 3,
        "db_id": "movies",
        "question": "q3",
        "SQL": "SELECT 3",
        "grading_method": "set",
    },
]


@pytest.fixture
def bundle(tmp_path) -> BirdDatasetLoader:
    questions = tmp_path / "bird.jsonl"
    questions.write_text("\n".join(json.dumps(r) for r in ROWS))
    databases = tmp_path / "databases"
    databases.mkdir()
    (databases / "shop.sqlite").touch()
    # question 2's gold SQL broke on Trino
    (databases / "verified_questions.json").write_text(
        json.dumps({"postgres": {"shop": [1, 2]}, "trino": {"shop": [1]}})
    )
    return BirdDatasetLoader(path=str(questions), databases_dir=str(databases))


# ── gold SQL ──────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_sqlite_gold_sql_is_adapted_to_the_loaded_schema():
    sql = adapt_sqlite_benchmark_sql(
        "WITH top AS (SELECT * FROM `Products` WHERE ProductName LIKE '%bike%') "
        'SELECT "P"."ProductName" FROM top AS "P" JOIN Suppliers s ON s.ID = "P".SupplierID',
        "bird_shop",
        "postgres",
    )
    assert "ILIKE '%bike%'" in sql  # SQLite LIKE ignores case
    unquoted = sql.replace('"', "")
    assert "bird_shop.products" in unquoted and "bird_shop.suppliers" in unquoted
    assert "bird_shop.top" not in unquoted  # CTEs stay as they are
    assert "ProductName" not in sql  # names lower-cased, even quoted ones
    # and it still reads as Trino, where the golden SQL runs on the agent's engine
    assert "LOWER(" in transpile_sql(sql, read="postgres", write="trino")


@pytest.mark.unit
def test_qualify_tables_keeps_explicit_schemas():
    sql = qualify_tables("SELECT * FROM other.t JOIN u ON t.id = u.id", "s", "trino")
    assert sql == "SELECT * FROM other.t JOIN s.u ON t.id = u.id"


@pytest.mark.unit
def test_runner_adapts_bundled_bird_gold_sql(sample_dataset_item):
    item = DatasetItem(
        id="bird-1",
        input={"query": "q"},
        expected_output={"sql": "SELECT Name FROM Customers WHERE City LIKE 'paris'"},
        metadata={"sql_dialect": "sqlite", "sql_schema": "bird_shop"},
    )
    ctx = EvaluationContext(
        dataset_item=item,
        run_id="r",
        query="q",
        expected_sql=item.expected_sql,
        allowed_tables=[],
        dialect="postgres",
    )
    EvaluationRunner._adapt_expected_sql(ctx)
    assert (
        ctx.expected_sql
        == "SELECT name FROM bird_shop.customers WHERE city ILIKE 'paris'"
    )
    assert ctx.metadata["original_expected_sql"] == item.expected_sql


# ── dataset ───────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_bundled_database_keeps_questions_verified_on_every_engine(bundle):
    items = bundle.load("shop")
    assert [i.id for i in items] == ["bird-1"]
    assert items[0].metadata["sql_schema"] == "bird_shop"
    # not bundled: all its questions, no schema
    assert "sql_schema" not in bundle.load("movies")[0].metadata
    assert [i.id for i in bundle.load_bundled()] == ["bird-1"]


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("connection", "dataset"),
    [
        ("bird", "bird-all"),
        ("BIRD benchmark", "bird-all"),
        ("bird_movies", "bird-movies"),
        ("shop", "bird-shop"),
    ],
)
async def test_bird_connection_runs_the_bundled_databases(bundle, connection, dataset):
    provider = BirdBenchmarkProvider(bundle)
    assert provider.dataset_for_connection(connection) == dataset
    assert provider.owns(dataset)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_bird_all_loads_every_bundled_database(bundle):
    items = await BirdBenchmarkProvider(bundle).load("bird-all")
    assert [i.id for i in items] == ["bird-1"]


@pytest.mark.unit
def test_non_bird_connection_is_a_clear_error(bundle):
    with pytest.raises(
        DatasetNotFoundError, match=r"doesn't hold a BIRD database.*bundled: shop"
    ):
        BirdBenchmarkProvider(bundle).dataset_for_connection("AdventureWorksDW")


# ── loading ───────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_tables_are_read_with_types_every_value_fits():
    con = sqlite3.connect(":memory:")
    con.executescript(
        """
        CREATE TABLE Office (officeCode INTEGER PRIMARY KEY, City TEXT, opened DATE);
        CREATE TABLE Staff (id INTEGER, office INTEGER REFERENCES Office(officeCode),
                            salary REAL, badge INTEGER);
        INSERT INTO Office VALUES (1, 'Paris', '2003-01-06');
        INSERT INTO Staff VALUES (7, 1, 1200.5, 'n/a');
        """
    )
    office, staff = read_tables(con)
    assert [(c.name, c.kind) for c in office.columns] == [
        ("officecode", "bigint"),
        ("city", "text"),
        ("opened", "text"),  # dates stay ISO text
    ]
    assert office.primary_key == ["officecode"]
    assert [(c.name, c.kind) for c in staff.columns] == [
        ("id", "bigint"),
        ("office", "bigint"),
        ("salary", "double"),
        ("badge", "text"),  # an INTEGER column holding text falls back to text
    ]
    assert staff.foreign_keys == [(["office"], "office", ["officecode"])]
    pk, fk = _key_statements("bird_x", [office, staff])
    assert pk == 'ALTER TABLE "bird_x"."office" ADD PRIMARY KEY ("officecode")'
    assert fk.endswith('REFERENCES "bird_x"."office" ("officecode") NOT VALID')


@pytest.mark.unit
@pytest.mark.parametrize(
    ("value", "kind", "literal"),
    [
        (None, "text", "NULL"),
        ("O'Hara", "text", "'O''Hara'"),
        (3, "bigint", "3"),
        (1.5, "double", "DOUBLE '1.5'"),
        (b"\x01\xff", "binary", "X'01ff'"),
    ],
)
def test_trino_literals(value, kind, literal):
    assert _trino_literal(value, kind) == literal

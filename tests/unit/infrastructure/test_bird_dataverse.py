"""
test_bird_dataverse.py — BIRD databases pushed to dataverse through Oasis
(push_bird_to_oasis.py): gold SQL rewritten to the dataverse names, which
must be the names the push script gives the sources, and the "bird" dataset
when BIRD_DATAVERSE_SCHEMA is set.
"""

from __future__ import annotations

import json
import sqlite3

import pytest

from src.domain.entities.dataset_item import DatasetItem
from src.domain.entities.evaluation_context import EvaluationContext
from src.domain.evaluators.sql_comparison_utils import adapt_sqlite_benchmark_sql
from src.domain.services.evaluation_runner import EvaluationRunner
from src.infrastructure.bird import (
    convert_bird_trino_questions as convert,
    push_bird_to_oasis as oasis,
)
from src.infrastructure.bird.bird_benchmark_provider import BirdBenchmarkProvider
from src.infrastructure.bird.bird_dataset_loader import BirdDatasetLoader
from src.infrastructure.bird.load_bird_databases import (
    bundle_types,
    find_databases,
    read_tables,
    use_clean_names,
)
from src.infrastructure.bird.verify_bird_questions import gold_sql_for

SCHEMA = "agency__bird-public-benchmark"


@pytest.fixture
def shop(tmp_path):
    """A BIRD-style SQLite database: mixed-case names, spaces and symbols."""
    path = tmp_path / "car_retails.sqlite"
    con = sqlite3.connect(path)
    con.execute(
        'CREATE TABLE "Customers" ("Customer ID" INTEGER, "Credit ($)" REAL, City TEXT)'
    )
    con.execute('CREATE TABLE "Orders" (id INTEGER, "Customer ID" INTEGER)')
    con.executemany(
        "INSERT INTO Customers VALUES (?, ?, ?)",
        [(1, 10.0, "Paris"), (2, 20.0, "Rome"), (3, 30.0, "paris")],
    )
    con.executemany("INSERT INTO Orders VALUES (?, ?)", [(1, 1), (2, 1), (3, 2)])
    con.commit()
    con.close()
    return path


def _as_pushed(source, tmp_path):
    """
    The database as dataverse holds it, in SQLite: every table named and
    typed as push_bird_to_oasis creates its source, in schema SCHEMA.
    """
    con = sqlite3.connect(":memory:")
    con.execute(f"ATTACH DATABASE '{tmp_path / 'dataverse.sqlite'}' AS \"{SCHEMA}\"")
    for plan in oasis.plan_database(source):
        fields = list(plan.columns.values())
        con.execute(
            f'CREATE TABLE "{SCHEMA}"."{plan.english_name}" ({", ".join(fields)})'
        )
        rows = [list(r.values()) for r in oasis.records(source, plan)]
        con.executemany(
            f'INSERT INTO "{SCHEMA}"."{plan.english_name}" '
            f"VALUES ({', '.join('?' * len(fields))})",
            rows,
        )
    return con


@pytest.mark.unit
@pytest.mark.parametrize(
    "gold",
    [
        "SELECT T1.`Customer ID`, T1.`Credit ($)` FROM Customers AS T1 WHERE T1.City LIKE 'paris'",
        # columns qualified by the table's own name, a join and a CTE
        "WITH big AS (SELECT * FROM Customers WHERE `Credit ($)` > 15) "
        "SELECT big.City, COUNT(Orders.id) FROM big JOIN Orders "
        "ON Orders.`Customer ID` = big.`Customer ID` GROUP BY big.City",
        "SELECT Customers.City, Orders.id FROM Customers, Orders "
        "WHERE Customers.`Customer ID` = Orders.`Customer ID` ORDER BY Orders.id",
    ],
)
def test_gold_sql_returns_the_same_on_the_pushed_tables(shop, tmp_path, gold):
    original = sqlite3.connect(shop)
    expected = sorted(original.execute(gold).fetchall())

    sql = adapt_sqlite_benchmark_sql(gold, SCHEMA, "sqlite", db_id="car_retails")
    assert f'"{SCHEMA}"."car-retails-' in sql
    actual = sorted(_as_pushed(shop, tmp_path).execute(sql).fetchall())
    assert actual == expected


@pytest.mark.unit
def test_gold_sql_for_trino_on_dataverse():
    sql = gold_sql_for(
        "SELECT `Customer ID` FROM Customers WHERE City LIKE 'a%'",
        SCHEMA,
        "trino",
        "car_retails",
    )
    assert sql == (
        f'SELECT "customer_id" FROM "{SCHEMA}"."car-retails-customers" AS customers '
        "WHERE LOWER(city) LIKE LOWER('a%')"
    )


@pytest.mark.unit
def test_runner_uses_the_dataverse_names():
    item = DatasetItem(
        id="bird-1",
        input={"query": "q"},
        expected_output={"sql": "SELECT `Customer ID` FROM Customers"},
        metadata={
            "sql_dialect": "sqlite",
            "sql_schema": SCHEMA,
            "sql_db_id": "car_retails",
        },
    )
    ctx = EvaluationContext(
        dataset_item=item,
        run_id="r",
        query="q",
        expected_sql=item.expected_sql,
        allowed_tables=[],
        dialect="trino",
    )
    EvaluationRunner._adapt_expected_sql(ctx)
    assert ctx.expected_sql == (
        f'SELECT "customer_id" FROM "{SCHEMA}"."car-retails-customers" AS customers'
    )


# ── dataset ───────────────────────────────────────────────────────────────────

ROWS = [
    {"question_id": 1, "db_id": "shop", "question": "q1", "SQL": "SELECT 1"},
    {"question_id": 2, "db_id": "shop", "question": "q2", "SQL": "SELECT 2"},
    {"question_id": 3, "db_id": "movie_platform", "question": "q3", "SQL": "SELECT 3"},
]


@pytest.fixture
def loader(tmp_path) -> BirdDatasetLoader:
    questions = tmp_path / "bird.jsonl"
    questions.write_text("\n".join(json.dumps(r) for r in ROWS))
    databases = tmp_path / "databases"
    databases.mkdir()
    (databases / "shop.sqlite").touch()
    # trino/postgres checks are for the bird_<db> schemas, not dataverse
    (databases / "verified_questions.json").write_text(
        json.dumps({"trino": {"shop": [1]}, "dataverse": {"shop": [2]}})
    )
    return BirdDatasetLoader(
        path=str(questions), databases_dir=str(databases), dataverse_schema=SCHEMA
    )


@pytest.mark.unit
def test_every_database_is_on_dataverse(loader):
    assert loader.bundled_db_ids() == ["movie_platform", "shop"]
    item = loader.load("movie_platform")[0]
    assert item.metadata["sql_schema"] == SCHEMA
    assert item.metadata["sql_db_id"] == "movie_platform"
    # only the questions verified on dataverse; unverified databases keep all
    assert [i.id for i in loader.load("shop")] == ["bird-2"]
    assert [i.id for i in loader.load_bundled()] == ["bird-3", "bird-2"]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_bird_on_the_pool_connection_runs_every_database(loader):
    provider = BirdBenchmarkProvider(loader)
    dataset = provider.dataset_for_connection("bird-public-benchmark")
    assert dataset == "bird-all"
    assert len(await provider.load(dataset)) == 2
    assert provider.dataset_for_connection("movie_platform") == "bird-movie_platform"


@pytest.mark.unit
def test_push_script_names_match_the_gold_sql(shop):
    customers = next(p for p in oasis.plan_database(shop) if p.table == "customers")
    assert customers.english_name == "car-retails-customers"
    assert list(customers.columns.values()) == ["customer_id", "credit", "city"]


# ── t2s_bird_trino_he bundle: Trino gold SQL and its databases ────────────────


@pytest.mark.unit
def test_trino_gold_sql_runs_on_the_pushed_tables(shop, tmp_path):
    """The bundle's trino_sql reads <db_id>.<table> with names like "customer id"."""
    original = sqlite3.connect(shop)
    gold = (
        'SELECT t1."customer id" AS "customer id", COUNT(orders.id) AS _col_1 '
        'FROM customers AS t1 JOIN orders ON orders."customer id" = t1."customer id" '
        "WHERE LOWER(t1.city) = 'paris' GROUP BY t1.\"customer id\""
    )
    expected = original.execute(
        'SELECT c."Customer ID", COUNT(o.id) FROM Customers c JOIN Orders o '
        'ON o."Customer ID" = c."Customer ID" WHERE LOWER(c.City) = \'paris\' '
        'GROUP BY c."Customer ID"'
    ).fetchall()

    row = {"db_id": "car_retails", "trino_sql": gold}
    sql = convert.target_sql(row, "dataverse", SCHEMA)
    assert f'"{SCHEMA}"."car-retails-customers" AS t1' in sql
    assert f'"{SCHEMA}"."car-retails-orders" AS orders' in sql
    assert '"customer_id"' in sql
    actual = _as_pushed(shop, tmp_path).execute(sql).fetchall()
    assert actual == expected


@pytest.mark.unit
def test_trino_gold_sql_for_postgres_keeps_the_names():
    row = {"db_id": "car_retails", "trino_sql": 'SELECT "customer id" FROM customers'}
    assert convert.target_sql(row, "postgres") == (
        'SELECT "customer id" FROM car_retails.customers'
    )


@pytest.mark.unit
def test_bundle_layout_and_types(tmp_path, shop):
    nested = tmp_path / "databases" / "car_retails"
    nested.mkdir(parents=True)
    shop.rename(nested / "car_retails.sqlite")
    (tmp_path / "databases" / "books.sqlite").touch()
    assert list(find_databases(tmp_path / "databases")) == ["books", "car_retails"]

    schemas = tmp_path / "schemas"
    schemas.mkdir()
    (schemas / "car_retails.json").write_text(
        json.dumps(
            {
                "tables": [
                    {
                        "name": "customers",
                        "columns": [
                            {"name": "customer id", "trino_type": "varchar"},
                            {"name": "credit ($)", "trino_type": "double"},
                        ],
                    }
                ]
            }
        )
    )
    types = bundle_types(schemas, "car_retails")
    assert bundle_types(schemas, "books") is None
    con = sqlite3.connect(nested / "car_retails.sqlite")
    customers = next(t for t in read_tables(con, types) if t.name == "customers")
    # the bundle's types win; columns it doesn't list are inferred as before
    assert [(c.name, c.kind) for c in customers.columns] == [
        ("customer id", "text"),
        ("credit ($)", "double"),
        ("city", "text"),
    ]


@pytest.mark.unit
def test_clean_names_match_the_push_script_and_rename_the_keys(tmp_path):
    path = tmp_path / "shop.sqlite"
    con = sqlite3.connect(path)
    con.execute('CREATE TABLE "Parent" ("Parent ID" INTEGER PRIMARY KEY)')
    con.execute(
        'CREATE TABLE child ("Total ($)" REAL, "Parent Ref" INTEGER '
        'REFERENCES Parent("Parent ID"))'
    )
    con.commit()

    tables = read_tables(con)
    use_clean_names(tables)
    plans = {p.table: p for p in oasis.plan_database(path)}
    for table in tables:
        assert [c.name for c in table.columns] == list(
            plans[table.name].columns.values()
        )
    by_name = {t.name: t for t in tables}
    assert by_name["parent"].primary_key == ["parent_id"]
    assert by_name["child"].foreign_keys == [(["parent_ref"], "parent", ["parent_id"])]

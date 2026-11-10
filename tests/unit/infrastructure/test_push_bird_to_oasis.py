# ruff: noqa: RUF001
import json
import sqlite3

import httpx
import pytest

from src.infrastructure.bird import push_bird_to_oasis as oasis


@pytest.fixture
def database(tmp_path):
    path = tmp_path / "shop.sqlite"
    con = sqlite3.connect(path)
    con.execute(
        'CREATE TABLE "Orders" ("Order ID" INTEGER, "Total ($)" REAL, note TEXT, pic BLOB)'
    )
    con.executemany(
        "INSERT INTO Orders VALUES (?, ?, ?, ?)",
        [(1, 9.5, "a", b"\x00\x01"), (2, None, None, None), (3, 1.0, "c", None)],
    )
    con.execute("CREATE TABLE customers (id INTEGER, name TEXT)")
    con.commit()
    con.close()
    return path


class FakeOasis:
    def __init__(self):
        self.created, self.pools, self.pushed = [], [], []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if request.url.path == "/metro/api/sources/push":
            assert request.headers["x-api-token"] == "api-token"
            self.created.append(body)
            return httpx.Response(200, json={"sourceId": f"src-{len(self.created)}"})
        if request.url.path.startswith("/catalog/api/pools/"):
            self.pools.append((request.url.path, body))
            return httpx.Response(200, json={})
        self.pushed.append((request.headers["Source-Access-Token"], body))
        return httpx.Response(200, json={})


def test_names_are_valid_for_avro_and_oasis():
    assert oasis.clean_name("Total ($)") == "total"
    assert oasis.clean_name("2019 Count") == "_2019_count"
    assert oasis.field_names(["a b", "a_b"]) == {"a b": "a_b", "a_b": "a_b_2"}
    assert oasis.hebrew_words("car_retails") == "קאר רטאילס"
    assert oasis.hebrew_number(2) == "ב'"
    assert oasis.hebrew_number(15) == 'ט"ו'
    assert oasis.hebrew_number(123) == 'קכ"ג'


def test_hebrew_names_are_unique_and_only_hebrew():
    plan = oasis.SourcePlan("db1", "t", "t", {}, [])
    used: set[str] = set()
    first, second = oasis.hebrew_name(plan, used), oasis.hebrew_name(plan, used)
    assert first == "בירד דב אחת ט"
    assert second == "בירד דב אחת ט ב'"


def test_create_then_push(database, tmp_path):
    fake = FakeOasis()
    client = httpx.Client(transport=httpx.MockTransport(fake))
    plans = oasis.plan_database(database)
    state = oasis.State(tmp_path / "state.json")

    oasis.create_sources(client, plans, state, "api-token", "pool-1")
    orders = fake.created[0]
    assert orders["is-flowing"] is True
    assert orders["properties"]["english-source-name"] == "shop-orders"
    schema = json.loads(orders["schema"]["schema"])
    assert [f["name"] for f in schema["fields"]] == ["order_id", "total", "note", "pic"]
    assert [f["type"][1] for f in schema["fields"]] == [
        "long",
        "double",
        "string",
        "string",
    ]
    assert "map" not in orders["schema"]["schema"]
    # Oasis's own schema format: id, path and originalName everywhere
    assert "namespace" not in schema and schema["path"] == ""
    assert schema["fields"][0] == {
        "path": "order_id",
        "originalName": "order_id",
        "name": "order_id",
        "id": schema["fields"][0]["id"],
        "type": ["null", "long"],
    }
    ids = [schema["id"], *(f["id"] for f in schema["fields"])]
    assert len(set(ids)) == len(ids)
    assert fake.pools == [("/catalog/api/pools/pool-1/sources", ["src-1", "src-2"])]
    assert state.sources["shop.orders"]["trino_table"] == (
        'dataverse."agency__bird-public-benchmark"."shop-orders"'
    )

    # Re-running creates nothing new
    oasis.create_sources(
        client, plans, oasis.State(tmp_path / "state.json"), "api-token", "pool-1"
    )
    assert len(fake.created) == 2 and len(fake.pools) == 1

    state = oasis.State(tmp_path / "state.json")
    missing = oasis.push_rows(
        client, {"shop": database}, plans, state, {"src-1": "tok-orders"}, batch_size=2
    )
    assert missing == ["shop.customers"]
    assert fake.pushed == [
        (
            "tok-orders",
            [
                {"order_id": 1, "total": 9.5, "note": "a", "pic": "AAE="},
                {"order_id": 2, "total": None, "note": None, "pic": None},
            ],
        ),
        ("tok-orders", [{"order_id": 3, "total": 1.0, "note": "c", "pic": None}]),
    ]
    assert state.sources["shop.orders"]["done"]


def test_push_continues_after_the_last_batch_sent(database, tmp_path):
    fake = FakeOasis()
    client = httpx.Client(transport=httpx.MockTransport(fake))
    plans = [p for p in oasis.plan_database(database) if p.table == "orders"]
    state = oasis.State(tmp_path / "state.json")
    oasis.create_sources(client, plans, state, "api-token", "pool-1")
    state.sources["shop.orders"]["pushed_rows"] = 2

    oasis.push_rows(client, {"shop": database}, plans, state, {"shop.orders": "t"}, 500)
    assert [len(batch) for _, batch in fake.pushed] == [1]


def test_destination_comes_from_a_file(database, tmp_path):
    plan = oasis.plan_database(database)[0]
    assert oasis.source_body(plan, "x")["destination"] == {
        "destinations": [{"type": "our-s3", "partition-field": "_metro_insertion_ts"}]
    }

    entry = {"type": "our-s3", "name": "bird"}
    as_list = tmp_path / "list.json"
    as_list.write_text(json.dumps([entry]))
    as_object = tmp_path / "object.json"
    as_object.write_text(json.dumps({"destinations": [entry]}))
    for path in (as_list, as_object):
        destination = oasis.read_destination(path)
        assert destination == {"destinations": [entry]}
        assert oasis.source_body(plan, "x", destination)["destination"] == destination

    fake = FakeOasis()
    client = httpx.Client(transport=httpx.MockTransport(fake))
    state = oasis.State(tmp_path / "state.json")
    oasis.create_sources(
        client,
        [plan],
        state,
        "api-token",
        "pool-1",
        destination={"destinations": [entry]},
    )
    assert fake.created[0]["destination"] == {"destinations": [entry]}


def test_a_name_oasis_already_has_is_reported_and_the_rest_are_created(
    database, tmp_path
):
    fake = FakeOasis()
    attempts = []

    def oasis_with_a_leftover(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/metro/api/sources/push":
            name = json.loads(request.content)["properties"]["english-source-name"]
            attempts.append(name)
            if name == "shop-customers":
                return httpx.Response(
                    422, json={"message": "english source name must be unique"}
                )
        return fake(request)

    client = httpx.Client(transport=httpx.MockTransport(oasis_with_a_leftover))
    plans = oasis.plan_database(database)
    state = oasis.State(tmp_path / "state.json")

    taken = oasis.create_sources(client, plans, state, "api-token", "pool-1")

    assert taken == ["shop-customers"]
    assert attempts.count("shop-customers") == 1
    assert list(state.sources) == ["shop.orders"]
    assert fake.pools == [("/catalog/api/pools/pool-1/sources", ["src-1"])]


def test_create_stops_on_other_errors_without_retrying(database, tmp_path):
    calls = []

    def broken(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(500, text="boom")

    client = httpx.Client(transport=httpx.MockTransport(broken))
    plans = oasis.plan_database(database)
    state = oasis.State(tmp_path / "state.json")
    with pytest.raises(RuntimeError, match="500"):
        oasis.create_sources(client, plans, state, "api-token", "pool-1")
    assert len(calls) == 1


def test_create_and_push_log_every_table(database, tmp_path, capsys):
    fake = FakeOasis()
    client = httpx.Client(transport=httpx.MockTransport(fake))
    plans = oasis.plan_database(database)
    state = oasis.State(tmp_path / "state.json")

    oasis.create_sources(client, plans, state, "api-token", "pool-1")
    out = capsys.readouterr().out
    assert "[1/2] shop.customers: creating shop-customers" in out or (
        "[1/2] shop.orders: creating shop-orders" in out
    )
    assert "created src-1 in " in out
    assert "pool: added 2 sources to pool pool-1" in out

    oasis.create_sources(client, plans, state, "api-token", "pool-1")
    assert "already created, skipping" in capsys.readouterr().out

    oasis.push_rows(client, {"shop": database}, plans, state, {"src-1": "t"}, 2)
    out = capsys.readouterr().out
    assert "no Source-Access-Token, skipping" in out
    assert "2/3 rows" in out and "3/3 rows" in out
    assert "push: finished, 1 tables not pushed" in out


def test_a_timeout_ends_with_one_clear_message(database, tmp_path, monkeypatch):
    def no_answer(*args, **kwargs):
        raise httpx.ReadTimeout("The read operation timed out")

    monkeypatch.setattr(oasis, "create_sources", no_answer)
    monkeypatch.setenv("OASIS_API_TOKEN", "t")
    monkeypatch.setenv("OASIS_POOL_ID", "p")
    with pytest.raises(SystemExit, match="did not answer within") as ended:
        oasis.run(
            [
                "create",
                "--databases-dir",
                str(tmp_path),
                "--state",
                str(tmp_path / "state.json"),
            ]
        )
    assert "delete it" in str(ended.value)


def test_tokens_file_can_be_a_list(tmp_path):
    path = tmp_path / "tokens.json"
    path.write_text(
        json.dumps(
            [
                {"sourceId": "id-1", "name": "shop-orders", "sourceAccessToken": " a "},
                {"source_id": "id-2", "token": "b"},
                ["shop.customers", "c"],
            ]
        )
    )
    assert oasis.read_tokens(path) == {
        "id-1": "a",
        "shop-orders": "a",
        "id-2": "b",
        "shop.customers": "c",
    }

    path.write_text(json.dumps([{"id": "x", "secret": "y"}]))
    with pytest.raises(SystemExit, match=r"fields: id, secret"):
        oasis.read_tokens(path)


def test_push_finds_the_token_by_source_name(database, tmp_path):
    fake = FakeOasis()
    client = httpx.Client(transport=httpx.MockTransport(fake))
    plans = [p for p in oasis.plan_database(database) if p.table == "orders"]
    state = oasis.State(tmp_path / "state.json")
    oasis.create_sources(client, plans, state, "api-token", "pool-1")
    missing = oasis.push_rows(
        client, {"shop": database}, plans, state, {"shop-orders": "t"}, 500
    )
    assert missing == [] and fake.pushed[0][0] == "t"


def test_tokens_list_with_name_and_source_key(tmp_path):
    path = tmp_path / "tokens.json"
    path.write_text(
        json.dumps(
            [
                {"name": "shop-orders", "source_key": "a"},
                {"name": "shop-customers", "source_key": "b"},
            ]
        )
    )
    assert oasis.read_tokens(path) == {"shop-orders": "a", "shop-customers": "b"}

    # only the database's name: can't tell its tables apart
    path.write_text(
        json.dumps(
            [{"name": "shop", "source_key": "a"}, {"name": "shop", "source_key": "b"}]
        )
    )
    with pytest.raises(SystemExit, match="two different tokens"):
        oasis.read_tokens(path)


def test_batches_stay_under_the_size_limit():
    rows = [{"x": "a" * 100} for _ in range(10)]
    row_size = len(json.dumps(rows[0])) + 1
    batches = list(oasis._sized_batches(iter(rows), 500, 3 * row_size + 2))
    assert [len(b) for b in batches] == [3, 3, 3, 1]
    assert [len(b) for b in oasis._sized_batches(iter(rows), 4, 10**6)] == [4, 4, 2]
    # a row bigger than the limit still goes, on its own
    assert [len(b) for b in oasis._sized_batches(iter(rows), 500, 10)] == [1] * 10
    assert all(len(json.dumps(b)) <= 3 * row_size + 2 for b in batches)

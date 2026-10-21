"""
test_bird_dataset.py — BIRD-Verified loading, connection → db_id mapping,
per-question grading and SQLite gold SQL adaptation.
"""

from __future__ import annotations

import json
import socket
from unittest.mock import AsyncMock

import pytest

from src.application.dto.run_mcp_dataset_request import RunMcpDatasetRequest
from src.application.use_cases.run_mcp_dataset_evaluation_use_case import (
    RunMcpDatasetEvaluationUseCase,
)
from src.domain.entities.dataset_item import DatasetItem
from src.domain.entities.evaluation_context import EvaluationContext
from src.domain.entities.query_result import QueryResult
from src.domain.evaluators.contains_evaluator import ContainsEvaluator
from src.domain.evaluators.sql_comparison_utils import (
    Grading,
    compare_results,
    parse_grading_method,
)
from src.domain.services.evaluation_runner import EvaluationRunner
from src.infrastructure.bird.bird_dataset_loader import (
    BirdDatasetLoader,
    bird_db_id_for_connection,
)
from src.infrastructure.langfuse.langfuse_dataset_provider import (
    DatasetNotFoundError,
    LangfuseDatasetProvider,
)

ROWS = [
    {
        "question_id": 11,
        "db_id": "movie_platform",
        "question": "Name the movies rated 4+ in 2020.",
        "evidence": "ratings in year 2020 refers to rating_timestamp_utc like '%2020%'",
        "SQL": "SELECT DISTINCT m.`movie_title` FROM `ratings` r JOIN `movies` m "
        "ON r.`movie_id` = m.`movie_id` WHERE SUBSTR(r.`rating_timestamp_utc`,1,4) = '2020'",
        "grading_method": "set",
    },
    {
        "question_id": 12,
        "db_id": "movie_platform",
        "question": "Name 3 movies.",
        "evidence": "",
        "SQL": "SELECT movie_title FROM movies",
        "grading_method": "subset,=,3",
    },
    {
        "question_id": 900,
        "db_id": "books",
        "question": "How many books?",
        "evidence": None,
        "SQL": "SELECT COUNT(*) FROM book",
        "grading_method": "set",
    },
]


@pytest.fixture
def loader(tmp_path) -> BirdDatasetLoader:
    path = tmp_path / "bird.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in ROWS))
    return BirdDatasetLoader(path=str(path))


def _qr(rows, columns=("v",), truncated=False) -> QueryResult:
    return QueryResult(success=True, rows=[list(r) for r in rows], columns=list(columns), truncated=truncated)


# ── loader ────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_loader_builds_items_per_database(loader):
    items = loader.load("movie_platform")
    assert [i.id for i in items] == ["bird-11", "bird-12"]
    first = items[0]
    assert first.query == (
        "Name the movies rated 4+ in 2020.\n\n"
        "Hint: ratings in year 2020 refers to rating_timestamp_utc like '%2020%'"
    )
    assert first.metadata["grading_method"] == "set"
    assert first.metadata["sql_dialect"] == "sqlite"
    assert items[1].query == "Name 3 movies."  # no hint when evidence is empty
    assert loader.db_ids() == ["books", "movie_platform"]
    assert loader.load("unknown") == []


@pytest.mark.unit
@pytest.mark.parametrize(
    ("name", "db_id"),
    [
        ("movie_platform", "movie_platform"),
        ("BIRD_movie_platform", "movie_platform"),
        ("bird-movie platform", "movie_platform"),
        ("", None),
        (None, None),
    ],
)
def test_db_id_from_connection_name(name, db_id):
    assert bird_db_id_for_connection(name) == db_id


@pytest.mark.unit
@pytest.mark.asyncio
async def test_unknown_bird_database_lists_available_ones(loader):
    provider = LangfuseDatasetProvider(client=None, parser=None, bird_loader=loader)
    with pytest.raises(DatasetNotFoundError, match="books, movie_platform"):
        await provider.get_dataset("bird-nope")


@pytest.mark.unit
@pytest.mark.asyncio
async def test_bird_items_are_synced_to_langfuse(loader):
    client = AsyncMock()
    client.create_dataset = lambda **kw: None
    created = []
    client.create_dataset_item = lambda **kw: created.append(kw["id"])
    provider = LangfuseDatasetProvider(client=client, parser=None, bird_loader=loader)

    items = await provider.get_dataset("bird-movie_platform")
    assert [i.id for i in items] == created == ["bird-11", "bird-12"]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_use_case_resolves_bird_from_connection_name():
    runner = AsyncMock(spec=EvaluationRunner)
    runner.run = AsyncMock(side_effect=RuntimeError("stop"))
    metadata = AsyncMock()
    metadata.get_connection_name = AsyncMock(return_value="bird_movie_platform")
    use_case = RunMcpDatasetEvaluationUseCase(runner=runner, metadata_provider=metadata)

    with pytest.raises(RuntimeError, match="stop"):
        await use_case.execute(RunMcpDatasetRequest(dataset_name="bird", connection_id=7))
    assert runner.run.await_args.kwargs["dataset_name"] == "bird-movie_platform"


@pytest.mark.unit
def test_bundled_dataset_loads_without_network(monkeypatch):
    """Private-network deployments: loading BIRD must never touch the internet."""
    from src.api.dependencies.container import _get_bird_loader

    def no_network(*args, **kwargs):
        raise AssertionError("network access attempted")

    monkeypatch.setattr(socket, "create_connection", no_network)
    monkeypatch.setattr(socket.socket, "connect", no_network)
    _get_bird_loader.cache_clear()

    bird = _get_bird_loader()
    assert len(bird.db_ids()) == 69
    items = [i for db in bird.db_ids() for i in bird.load(db)]
    assert len(items) == 2462
    assert len({i.id for i in items}) == 2462
    assert all(i.expected_sql and i.metadata["grading_method"] for i in items)


# ── grading ───────────────────────────────────────────────────────────────────


@pytest.mark.unit
@pytest.mark.parametrize(
    ("raw", "grading"),
    [
        ("set", Grading("set")),
        (" multiset", Grading("multiset")),
        ("list", Grading("list")),
        ("subset,=,3", Grading("subset", "=", 3)),
        ("subset,>=,1", Grading("subset", ">=", 1)),
        ("subset", Grading("subset")),
        ("", Grading()),
        (None, Grading()),
    ],
)
def test_parse_grading_method(raw, grading):
    assert parse_grading_method(raw) == grading


@pytest.mark.unit
def test_set_grading_ignores_duplicates_and_order():
    score, _ = compare_results(
        _qr([[1], [2], [2]]), _qr([[2], [1]]), allow_extra_columns=True,
        requires_ordering=True, grading=Grading("set"),
    )
    assert score == 1.0


@pytest.mark.unit
def test_list_grading_enforces_order():
    score, details = compare_results(
        _qr([[1], [2]]), _qr([[2], [1]]), allow_extra_columns=True, grading=Grading("list")
    )
    assert score == 0.0
    assert details["reason"] == "row_order_mismatch"


@pytest.mark.unit
@pytest.mark.parametrize(
    ("generated", "expected_score"),
    [
        ([["a"], ["b"], ["c"]], 1.0),  # any 3 valid answers
        ([["d"], ["b"], ["a"]], 1.0),
        ([["a"], ["b"]], 0.0),  # too few
        ([["a"], ["b"], ["x"]], 0.0),  # x is not a valid answer
    ],
)
def test_subset_grading(generated, expected_score):
    expected = _qr([["a"], ["b"], ["c"], ["d"]], ["title"])
    score, _ = compare_results(
        expected, _qr(generated, ["name"]), allow_extra_columns=True, grading=Grading("subset", "=", 3)
    )
    assert score == expected_score


@pytest.mark.unit
@pytest.mark.asyncio
async def test_contains_uses_the_items_grading_method(sample_dataset_item):
    item = DatasetItem(
        id="bird-12", input={"query": "q"}, expected_output={"sql": "SELECT movie_title FROM movies"},
        metadata={"grading_method": "subset,=,1"},
    )
    ctx = EvaluationContext(
        dataset_item=item, run_id="r", query="q", expected_sql="SELECT movie_title FROM movies",
        allowed_tables=[],
    )
    ctx.generated_sql = "SELECT movie_title FROM movies LIMIT 1"
    ctx.expected_result = _qr([["a"], ["b"]], ["movie_title"])
    ctx.generated_result = _qr([["b"]], ["movie_title"])
    result = await ContainsEvaluator().evaluate(ctx)
    assert result.score == 1.0


# ── gold SQL adaptation ───────────────────────────────────────────────────────


@pytest.mark.unit
def test_sqlite_gold_sql_is_transpiled_for_the_connection(loader):
    item = loader.load("movie_platform")[0]
    ctx = EvaluationContext(
        dataset_item=item, run_id="r", query=item.query, expected_sql=item.expected_sql,
        allowed_tables=[], dialect="postgres",
    )
    EvaluationRunner._adapt_expected_sql(ctx)
    assert "`" not in ctx.expected_sql
    assert '"ratings"' in ctx.expected_sql
    assert ctx.metadata["original_expected_sql"] == item.expected_sql

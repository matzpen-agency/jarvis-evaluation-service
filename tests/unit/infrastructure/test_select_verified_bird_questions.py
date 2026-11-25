import json

import pytest

from src.infrastructure.bird.select_verified_bird_questions import select


@pytest.fixture
def bundle(tmp_path):
    (tmp_path / "splits").mkdir()
    (tmp_path / "schemas").mkdir()
    rows = [
        # same question, same gold (formatting aside)
        {
            "db_id": "shop",
            "question_en": "How many orders?",
            "sqlite_sql": "select count(*) from `orders`",
            "trino_sql": "T1",
        },
        # same question, BIRD-Verified fixed the gold
        {
            "db_id": "shop",
            "question_en": "Name of the state?",
            "sqlite_sql": "SELECT state FROM t",
            "trino_sql": "T2",
        },
        # reworded by BIRD-Verified: the Hebrew is of the old wording
        {
            "db_id": "shop",
            "question_en": "Population in 2010?",
            "sqlite_sql": "SELECT 1",
            "trino_sql": "T3",
        },
        # not verified at all
        {
            "db_id": "shop",
            "question_en": "Unrelated?",
            "sqlite_sql": "SELECT 2",
            "trino_sql": "T4",
        },
    ]
    (tmp_path / "splits" / "train.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows)
    )
    (tmp_path / "schemas" / "shop.json").write_text(
        json.dumps(
            {
                "tables": [
                    {
                        "name": "t",
                        "columns": [{"name": "name", "trino_type": "varchar"}],
                    }
                ]
            }
        )
    )
    return tmp_path


VERIFIED = [
    {
        "question_id": 10,
        "db_id": "shop",
        "question": "How many orders ?",
        "SQL": "SELECT COUNT(*) FROM orders;",
        "grading_method": "set",
    },
    {
        "question_id": 11,
        "db_id": "shop",
        "question": "Name of the state?",
        "SQL": "SELECT name FROM t",
        "grading_method": "list",
    },
    {
        "question_id": 12,
        "db_id": "shop",
        "question": "Population in Georgia in 2010?",
        "SQL": "SELECT 1",
        "grading_method": "set",
    },
]


@pytest.mark.unit
def test_keeps_verified_questions_with_the_verified_gold(bundle):
    calls = []

    def to_trino(sql, schema):
        calls.append((sql, schema))
        return "CONVERTED"

    selected, counts = select(bundle, VERIFIED, to_trino)
    assert [r["bird_question_id"] for r in selected] == [10, 11]
    same, fixed = selected
    assert same["gold_source"] == "bundle" and same["trino_sql"] == "T1"
    assert fixed["gold_source"] == "bird_verified"
    assert fixed["sqlite_sql"] == "SELECT name FROM t"
    assert fixed["original_sqlite_sql"] == "SELECT state FROM t"
    assert fixed["trino_sql"] == "CONVERTED"
    assert fixed["grading_method"] == "list"
    assert calls == [("SELECT name FROM t", {"t": {"name": "varchar"}})]
    assert counts == {"bundle": 1, "bird_verified": 1}

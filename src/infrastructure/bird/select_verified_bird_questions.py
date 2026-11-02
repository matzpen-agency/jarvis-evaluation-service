"""
select_verified_bird_questions.py — Keep the t2s_bird_trino_he questions that
BIRD-Verified (wenyupapa/BIRD-Verified-CoT-2462-GPT5.4, bundled as
src/config/bird/bird_verified_cot_2462.jsonl) verified, with its gold SQL.

BIRD-Verified is 2462 BIRD train questions whose gold SQL was checked, and
fixed where it was wrong. A translated question is kept when its database and
English question are the same as a verified one's (case and punctuation
aside). Questions BIRD-Verified reworded are left out: the Hebrew was
translated from the old wording. Where the verified gold SQL differs from the
one the bundle converted, it replaces `sqlite_sql`, and `trino_sql` is
converted again with the bundle's own converter (code/src/t2s/sqlite_trino.py)
and the column types in its schemas/; `gold_source` is then "bird_verified",
else "bundle". Each question also gets BIRD-Verified's `bird_question_id` and
`grading_method`.

    python -m src.infrastructure.bird.select_verified_bird_questions \\
        --bundle path/to/t2s_bird_trino_he --out bird_verified_he.jsonl

Then convert_bird_trino_questions.py --verify checks every question's Trino
gold against the (verified) SQLite gold where the data is loaded.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
from collections import Counter
from pathlib import Path

import sqlglot

from src.infrastructure.bird.load_bird_databases import DATABASES_DIR, _say

VERIFIED_PATH = DATABASES_DIR.parent / "bird_verified_cot_2462.jsonl"
SPLITS = ("train", "validation", "dev")


def question_key(db_id: str, question: str) -> tuple[str, str]:
    return db_id, re.sub(r"[^a-z0-9]+", " ", (question or "").lower()).strip()


def _canonical(sql: str) -> str:
    tree = sqlglot.parse_one(sql, read="sqlite")
    for identifier in tree.find_all(sqlglot.exp.Identifier):
        # SQLite names ignore case and quoting
        identifier.set("this", identifier.name.lower())
        identifier.set("quoted", False)
    return tree.sql(dialect="sqlite", normalize=True)


def same_sql(a: str, b: str) -> bool:
    """Equal once parsed, so formatting, quoting and case don't count."""
    try:
        return _canonical(a) == _canonical(b)
    except Exception:
        return a.strip() == b.strip()


def load_converter(bundle: Path):
    """The bundle's to_trino(sql, schema), which made its trino_sql."""
    path = bundle / "code" / "src" / "t2s" / "sqlite_trino.py"
    spec = importlib.util.spec_from_file_location("t2s_sqlite_trino", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.to_trino


def trino_schema(bundle: Path, db_id: str) -> dict | None:
    """{table: {column: trino_type}}, as the bundle's converter takes it."""
    path = bundle / "schemas" / f"{db_id}.json"
    if not path.exists():
        return None
    return {
        t["name"]: {c["name"]: c["trino_type"] for c in t["columns"]}
        for t in json.loads(path.read_text(encoding="utf-8"))["tables"]
    }


def select(bundle: Path, verified: list[dict], to_trino) -> tuple[list[dict], Counter]:
    by_key = {question_key(v["db_id"], v["question"]): v for v in verified}
    counts: Counter = Counter()
    selected, schemas = [], {}
    for split in SPLITS:
        path = bundle / "splits" / f"{split}.jsonl"
        if not path.exists():
            continue
        for line in path.open(encoding="utf-8"):
            row = json.loads(line)
            v = by_key.get(question_key(row["db_id"], row["question_en"]))
            if v is None:
                continue
            row["split"] = split
            row["bird_question_id"] = v["question_id"]
            row["grading_method"] = v.get("grading_method") or ""
            row["gold_source"] = "bundle"
            if not same_sql(v["SQL"], row["sqlite_sql"]):
                if row["db_id"] not in schemas:
                    schemas[row["db_id"]] = trino_schema(bundle, row["db_id"])
                try:
                    trino_sql = to_trino(v["SQL"], schemas[row["db_id"]])
                except Exception as exc:
                    counts["verified SQL doesn't convert"] += 1
                    _say(f"  {row['db_id']} #{v['question_id']}: {exc!s:.120}")
                    continue
                row["original_sqlite_sql"] = row["sqlite_sql"]
                row["sqlite_sql"] = v["SQL"]
                row["trino_sql"] = trino_sql
                row["gold_source"] = "bird_verified"
            counts[row["gold_source"]] += 1
            selected.append(row)
    return selected, counts


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--bundle", type=Path, required=True, help="t2s_bird_trino_he folder"
    )
    parser.add_argument("--verified", type=Path, default=VERIFIED_PATH)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    verified = [json.loads(line) for line in args.verified.open(encoding="utf-8")]
    selected, counts = select(args.bundle, verified, load_converter(args.bundle))
    with args.out.open("w", encoding="utf-8") as out:
        for row in selected:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
    _say(
        f"{len(selected)} of {len(verified)} verified questions are translated → {args.out}\n"
        f"  gold SQL as in the bundle: {counts['bundle']}, "
        f"replaced by the verified one: {counts['bird_verified']}, "
        f"verified SQL didn't convert: {counts["verified SQL doesn't convert"]}"
    )


if __name__ == "__main__":
    main()

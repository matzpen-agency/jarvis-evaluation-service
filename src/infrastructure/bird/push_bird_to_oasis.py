"""
push_bird_to_oasis.py — Put BIRD SQLite databases into dataverse through Oasis.

Every table of every BIRD database becomes one Oasis push source (an Avro
record schema, no map types), all of them in the pool "bird-public-benchmark".
Dataverse then serves each one as
    dataverse."agency__bird-public-benchmark".<db_id>__<table>
with lower-case column names in which anything that isn't a letter, a digit
or "_" became "_" (Avro field names allow nothing else).

Two steps, because a source's Source-Access-Token can only be copied from the
Oasis UI:

  1. create  — creates the push sources and adds them to the pool:
       python -m src.infrastructure.bird.push_bird_to_oasis create --databases-dir /data/bird
  2. push    — sends every table's rows to its source, with the tokens in a
               JSON file {"<sourceId>": "<token>", ...} (keys may also be
               "<db_id>.<table>"), or a CSV file with columns source_id,token:
       python -m src.infrastructure.bird.push_bird_to_oasis push --databases-dir /data/bird --tokens tokens.json

--databases-dir is a folder of <db_id>.sqlite files or of
<db_id>/<db_id>.sqlite folders (the t2s_bird_trino_he bundle's databases/).
Give that bundle's schemas/ folder as --schemas-dir so the column types are
the ones its Trino gold SQL was converted for.

Fill in X_API_TOKEN and POOL_ID below (or set OASIS_API_TOKEN / OASIS_POOL_ID;
OASIS_DATA_FACTORY / OASIS_PROJECT override the data factory and project ids)
and don't commit them. Both steps can be re-run: what was done is kept in the
state file (--state, oasis_bird_sources.json), so sources aren't created twice
and pushing continues from the last batch sent. The state file also maps every
BIRD table and column to its dataverse name. The sources go to
Oasis's our-s3 destination (DESTINATION); --destination takes a JSON file to
send another one. --dry-run prints the requests without sending anything.
"""

# ruff: noqa: RUF001, RUF002 — Hebrew letters are intended here

from __future__ import annotations

import argparse
import base64
import csv
import json
import math
import os
import re
import sys
import time
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from src.domain.evaluators.sql_comparison_utils import (
    clean_name,
    dataverse_table_name,
)
from src.infrastructure.bird.load_bird_databases import (
    DATABASES_DIR,
    _batches,
    _original_names,
    _say,
    bundle_types,
    convert,
    find_databases,
    open_database,
    read_tables,
)

# ── Fill these in (never commit them) ─────────────────────────────────────────
X_API_TOKEN = ""
POOL_ID = ""
# ──────────────────────────────────────────────────────────────────────────────

OASIS_URL = "https://oasis.idf.cts"
PUSH_URL = "https://push-data-metro-pipeline.south.metro.idf.cts/api/v1/push"
POOL_NAME = "bird-public-benchmark"
TRINO_SCHEMA = f"agency__{POOL_NAME}"

# The project must belong to the data factory, or Oasis answers 422
# "could not find project in data factory"
DATA_FACTORY = os.environ.get(
    "OASIS_DATA_FACTORY", "ecc92565-16cd-44e0-b3ac-083e11165c72"
)
PROJECT = os.environ.get("OASIS_PROJECT", "152def03-d7e7-489e-bac5-a286f9e15c4f")
PROPERTIES = {
    "data-permission": {
        "permission-type": "restricted",
        "source-shared-data-factories": [{"id": DATA_FACTORY, "name": "string"}],
    },
    "data-factory": DATA_FACTORY,
    "project": PROJECT,
    "operational-contact": "agency",
    "operational-data-factory": "agency",
    "site": "south",
}
# Where the source's data goes; dataverse reads it from Oasis's own S3.
# Oasis wants the partition field to be one of the schema's fields, so every
# table gets it as an extra field, set to the push time on every row.
PARTITION_FIELD = "_metro_insersion_ts"
DESTINATION = {
    "destinations": [
        {"type": "our-s3", "partition-field": PARTITION_FIELD},
    ]
}
# Its Avro type, as JSON; OASIS_PARTITION_TYPE='"string"' sends an ISO time
# string and '"long"' plain epoch milliseconds instead
PARTITION_TYPE = json.loads(
    os.environ.get(
        "OASIS_PARTITION_TYPE", '{"type": "long", "logicalType": "timestamp-millis"}'
    )
)
ALERTS = {
    "alert-types": ["error"],
    "alerts-interval": "5",
    "chat-alerts": [{"contact": "s92232591@idfcrs.il"}],
    "email-alerts": [{"contact": "s92232591@idfcrs.il"}],
    "phone-alerts": [
        {"phone-number": "0556606939", "group": "agency", "contact-name": "yodan"}
    ],
}

# Column kinds of load_bird_databases → Avro types (blobs are sent as base64)
_AVRO_TYPES = {
    "bigint": "long",
    "double": "double",
    "text": "string",
    "binary": "string",
}


@dataclass
class SourcePlan:
    db_id: str
    table: str  # lower-case BIRD table name
    original_table: str  # as written in the SQLite file
    columns: dict[str, str]  # lower-case BIRD column → Avro field name
    kinds: list[str]  # column kinds, in SQLite column order

    @property
    def key(self) -> str:
        return f"{self.db_id}.{self.table}"

    @property
    def english_name(self) -> str:
        return dataverse_table_name(self.db_id, self.table)

    @property
    def record_name(self) -> str:
        # Avro type names allow no hyphens, unlike the kebab-case source name
        return f"{clean_name(self.db_id)}_{clean_name(self.table)}"

    @property
    def trino_table(self) -> str:
        return f'dataverse."{TRINO_SCHEMA}"."{self.english_name}"'

    def avro_schema(self) -> dict:
        """
        The schema in Oasis's own format (as its API returns a source's schema):
        every record and field has an id, a path and an originalName; no
        namespace and no defaults. Ids are fixed per table/column, so a dry run
        shows what create sends.
        """
        return {
            "type": "record",
            "name": self.record_name,
            "doc": "",
            "originalName": self.record_name,
            "id": _schema_id(self.key),
            "path": "",
            "fields": [
                {
                    "path": field,
                    "originalName": field,
                    "name": field,
                    "id": _schema_id(f"{self.key}.{field}"),
                    "type": ["null", _AVRO_TYPES[kind]],
                }
                for field, kind in zip(self.columns.values(), self.kinds, strict=True)
            ]
            + [
                {
                    "path": PARTITION_FIELD,
                    "originalName": PARTITION_FIELD,
                    "name": PARTITION_FIELD,
                    "id": _schema_id(f"{self.key}.{PARTITION_FIELD}"),
                    "type": PARTITION_TYPE,
                }
            ],
        }


def _schema_id(name: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"bird-public-benchmark/{name}"))


# ── Names ─────────────────────────────────────────────────────────────────────


def field_names(columns: list[str]) -> dict[str, str]:
    """BIRD column → unique Avro field name."""
    fields: dict[str, str] = {}
    used: set[str] = set()
    for column in columns:
        field, n = clean_name(column), 2
        while field in used:
            field, n = f"{clean_name(column)}_{n}", n + 1
        used.add(field)
        fields[column] = field
    return fields


_DIGRAPHS = [("sh", "ש"), ("ch", "צ'"), ("th", "ת'"), ("ph", "פ"), ("ck", "ק"),
             ("oo", "ו"), ("ee", "י")]  # fmt: skip
_LETTERS = dict(zip("abcdfghijklmnopqrstuvwxyz",
                    ["א", "ב", "ק", "ד", "פ", "ג", "ה", "י", "ג'", "ק", "ל", "מ", "נ",
                     "ו", "פ", "ק", "ר", "ס", "ט", "ו", "ו", "ו", "קס", "י", "ז"],
                    strict=True))  # fmt: skip
_DIGITS = ["אפס", "אחת", "שתיים", "שלוש", "ארבע", "חמש", "שש", "שבע", "שמונה", "תשע"]
_FINALS = {"כ": "ך", "מ": "ם", "נ": "ן", "פ": "ף", "צ": "ץ"}
_HEBREW_NAME = re.compile(r"""^[א-ת."' ]+$""")


def _transliterate(word: str) -> str:
    out, i = "", 0
    while i < len(word):
        pair = next((h for d, h in _DIGRAPHS if word.startswith(d, i)), None)
        if pair:
            out, i = out + pair, i + 2
            continue
        ch = word[i]
        out += ("א" if i == 0 else "") if ch == "e" else _LETTERS.get(ch, "")
        i += 1
    if out and out[-1] in _FINALS:
        out = out[:-1] + _FINALS[out[-1]]
    return out


def hebrew_words(name: str) -> str:
    """An English identifier written in Hebrew letters, e.g. car_retails → קאר רטילס."""
    words = []
    for word in re.findall(r"[a-z]+|[0-9]", name.lower()):
        words.append(_DIGITS[int(word)] if word.isdigit() else _transliterate(word))
    return " ".join(w for w in words if w)


def hebrew_number(n: int) -> str:
    """n in Hebrew letters (gematria), e.g. 15 → ט"ו, 2 → ב'."""
    ones, tens = "אבגדהוזחט", "יכלמנסעפצ"
    letters = "ת" * (n // 400) + ("", "ק", "ר", "ש")[(n % 400) // 100]
    rest = n % 100
    if rest in (15, 16):
        letters += "ט" + "וז"[rest - 15]
    else:
        letters += (tens[rest // 10 - 1] if rest >= 10 else "") + (
            ones[rest % 10 - 1] if rest % 10 else ""
        )
    return letters + "'" if len(letters) == 1 else f'{letters[:-1]}"{letters[-1]}'


def hebrew_name(plan: SourcePlan, used: set[str]) -> str:
    """A unique Hebrew name: only Hebrew letters, dots, quotes and inner spaces."""
    base = " ".join(
        f"בירד {hebrew_words(plan.db_id)} {hebrew_words(plan.table)}".split()
    )
    name, n = base, 2
    while name in used:
        name, n = f"{base} {hebrew_number(n)}", n + 1
    if not _HEBREW_NAME.match(name):
        raise ValueError(f"not a valid Hebrew source name: {name!r}")
    used.add(name)
    return name


# ── Reading the databases ─────────────────────────────────────────────────────


def plan_database(
    path: Path, types: dict[str, dict[str, str]] | None = None, db_id: str | None = None
) -> list[SourcePlan]:
    con = open_database(path)
    try:
        originals = _original_names(con)
        return [
            SourcePlan(
                db_id=db_id or path.stem,
                table=t.name,
                original_table=originals[t.name],
                columns=field_names([c.name for c in t.columns]),
                kinds=[c.kind for c in t.columns],
            )
            for t in read_tables(con, types)
        ]
    finally:
        con.close()


def _value(value: Any, kind: str) -> Any:
    value = convert(value, kind)
    if kind == "binary" and value is not None:
        return base64.b64encode(value).decode("ascii")
    if isinstance(value, float) and not math.isfinite(value):
        return None  # JSON has no NaN/Infinity
    return value


def records(
    path: Path, plan: SourcePlan, skip: int = 0, pushed_at: Any = None
) -> Iterator[dict]:
    """The table's rows as Avro records; pushed_at fills PARTITION_FIELD."""
    con = open_database(path)
    try:
        fields = list(plan.columns.values())
        for row in con.execute(
            f'SELECT * FROM "{plan.original_table}" LIMIT -1 OFFSET ?', (skip,)
        ):
            record = {
                f: _value(v, k) for f, v, k in zip(fields, row, plan.kinds, strict=True)
            }
            if pushed_at is not None:
                record[PARTITION_FIELD] = pushed_at
            yield record
    finally:
        con.close()


def row_count(path: Path, plan: SourcePlan) -> int:
    con = open_database(path)
    try:
        return con.execute(f'SELECT COUNT(*) FROM "{plan.original_table}"').fetchone()[
            0
        ]
    finally:
        con.close()


# ── Oasis ─────────────────────────────────────────────────────────────────────


def source_body(plan: SourcePlan, hebrew: str, destination: dict | None = None) -> dict:
    return {
        "is-flowing": True,
        "properties": {
            **PROPERTIES,
            "hebrew-source-name": hebrew,
            "english-source-name": plan.english_name,
            "description": f"BIRD benchmark, database {plan.db_id}, table {plan.table}",
        },
        "schema": {
            "schema": json.dumps(plan.avro_schema(), ensure_ascii=False),
            "data-structure": "STRUCTURED",
        },
        "destination": destination or DESTINATION,
        "transformations": {"transformations": []},
        "alerts": ALERTS,
    }


class State:
    """What was already done, kept in a JSON file so a re-run continues."""

    def __init__(self, path: Path):
        self.path = path
        self.sources: dict[str, dict] = (
            json.loads(path.read_text(encoding="utf-8"))["sources"]
            if path.exists()
            else {}
        )

    def save(self) -> None:
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps({"sources": self.sources}, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        tmp.replace(self.path)


def _post(client: httpx.Client, url: str, body: Any, headers: dict) -> httpx.Response:
    for attempt in range(4):
        try:
            response = client.post(url, json=body, headers=headers)
        except httpx.TransportError as exc:
            if attempt == 3:
                raise
            _say(f"    {exc!r}, retrying")
        else:
            if response.status_code < 500 or attempt == 3:
                if response.is_error:
                    raise RuntimeError(
                        f"POST {url} → {response.status_code}: {response.text[:1000]}"
                    )
                return response
            _say(f"    {response.status_code}, retrying")
        time.sleep(2**attempt)
    raise AssertionError("unreachable")


def _show(url: str, body: Any) -> None:
    _say(f"POST {url}\n{json.dumps(body, indent=2, ensure_ascii=False)}")


def create_sources(
    client: httpx.Client,
    plans: list[SourcePlan],
    state: State,
    token: str,
    pool_id: str,
    dry_run: bool = False,
    destination: dict | None = None,
) -> None:
    headers = {"x-api-token": token}
    used = {s["hebrew_name"] for s in state.sources.values()}
    for plan in plans:
        if plan.key in state.sources:
            continue
        body = source_body(plan, hebrew_name(plan, used), destination)
        if dry_run:
            _show(f"{OASIS_URL}/metro/api/sources/push", body)
            continue
        response = _post(client, f"{OASIS_URL}/metro/api/sources/push", body, headers)
        source_id = response.json()["sourceId"]
        state.sources[plan.key] = {
            "source_id": source_id,
            "english_name": plan.english_name,
            "hebrew_name": body["properties"]["hebrew-source-name"],
            "trino_table": plan.trino_table,
            "columns": plan.columns,
            "in_pool": False,
            "pushed_rows": 0,
            "done": False,
        }
        state.save()
        _say(f"created {plan.key} → {source_id}")

    new = [s["source_id"] for s in state.sources.values() if not s["in_pool"]]
    if not new:
        return
    url = f"{OASIS_URL}/catalog/api/pools/{pool_id}/sources"
    if dry_run:
        _show(url, new)
        return
    _post(client, url, new, headers)
    for s in state.sources.values():
        s["in_pool"] = True
    state.save()
    _say(f"added {len(new)} sources to pool {pool_id}")


def push_time() -> Any:
    """Now, as PARTITION_TYPE wants it."""
    if PARTITION_TYPE == "string":
        return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    return int(time.time() * 1000)


def push_rows(
    client: httpx.Client,
    paths: dict[str, Path],
    plans: list[SourcePlan],
    state: State,
    tokens: dict[str, str],
    batch_size: int,
    dry_run: bool = False,
) -> list[str]:
    """Push every table's rows; returns the tables skipped for a missing token."""
    missing = []
    for plan in plans:
        source = state.sources.get(plan.key)
        if source is None:
            missing.append(f"{plan.key} (no source yet, run create first)")
            continue
        if source["done"]:
            continue
        token = tokens.get(source["source_id"]) or tokens.get(plan.key)
        if not token and not dry_run:
            missing.append(plan.key)
            continue
        path = paths[plan.db_id]
        total = row_count(path, plan)
        rows = records(path, plan, source["pushed_rows"], push_time())
        for batch in _batches(rows, batch_size):
            if dry_run:
                _show(PUSH_URL, batch[:2])
                break
            _post(client, PUSH_URL, batch, {"Source-Access-Token": token})
            source["pushed_rows"] += len(batch)
            state.save()
            _say(f"{plan.key}: {source['pushed_rows']}/{total} rows")
        if not dry_run:
            source["done"] = True
            state.save()
    return missing


def read_destination(path: Path) -> dict:
    """
    The source's "destination" from a JSON file: either the whole object
    {"destinations": [...]} or just the list of destinations.
    """
    value = json.loads(path.read_text(encoding="utf-8"))
    return {"destinations": value} if isinstance(value, list) else value


def read_tokens(path: Path) -> dict[str, str]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".csv":
        return {
            r["source_id"].strip(): r["token"].strip()
            for r in csv.DictReader(text.splitlines())
        }
    return {k: str(v).strip() for k, v in json.loads(text).items()}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("step", choices=("create", "push"))
    parser.add_argument(
        "databases", nargs="*", help="db_ids (default: every .sqlite file)"
    )
    parser.add_argument("--databases-dir", type=Path, default=DATABASES_DIR)
    parser.add_argument(
        "--schemas-dir",
        type=Path,
        help="t2s_bird_trino_he schemas/ folder: take the column types from it",
    )
    parser.add_argument("--state", type=Path, default=Path("oasis_bird_sources.json"))
    parser.add_argument(
        "--tokens", type=Path, help="push: Source-Access-Token per source"
    )
    parser.add_argument(
        "--batch-size", type=int, default=500, help="push: rows per request"
    )
    parser.add_argument(
        "--ca-bundle", help="CA certificate file for the internal HTTPS"
    )
    parser.add_argument("--insecure", action="store_true", help="don't verify HTTPS")
    parser.add_argument(
        "--destination",
        type=Path,
        help="create: JSON file with another destination than our-s3",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="print the requests only"
    )
    args = parser.parse_args(argv)

    found = find_databases(args.databases_dir)
    db_ids = args.databases or list(found)
    if absent := sorted(set(db_ids) - set(found)):
        parser.error(f"not in {args.databases_dir}: {', '.join(absent)}")
    paths = {d: found[d] for d in db_ids}
    plans = [
        plan
        for d in db_ids
        for plan in plan_database(paths[d], bundle_types(args.schemas_dir, d), d)
    ]
    _say(f"{len(db_ids)} databases, {len(plans)} tables")

    state = State(args.state)
    verify: bool | str = False if args.insecure else (args.ca_bundle or True)
    with httpx.Client(verify=verify, timeout=120) as client:
        if args.step == "create":
            token = os.environ.get("OASIS_API_TOKEN", X_API_TOKEN)
            pool_id = os.environ.get("OASIS_POOL_ID", POOL_ID)
            if not args.dry_run and not (token and pool_id):
                parser.error(
                    "set X_API_TOKEN and POOL_ID in the script (or OASIS_API_TOKEN / OASIS_POOL_ID)"
                )
            destination = (
                read_destination(args.destination) if args.destination else None
            )
            create_sources(
                client,
                plans,
                state,
                token,
                pool_id or "<POOL_ID>",
                args.dry_run,
                destination,
            )
        else:
            if not args.tokens and not args.dry_run:
                parser.error("push needs --tokens")
            tokens = read_tokens(args.tokens) if args.tokens else {}
            missing = push_rows(
                client, paths, plans, state, tokens, args.batch_size, args.dry_run
            )
            if missing:
                _say("Not pushed, no token for:\n  " + "\n  ".join(missing))
                sys.exit(1)
    _say(f"State saved in {args.state}")


if __name__ == "__main__":
    main()

# BIRD-Verified-CoT-2462 (bundled subset)

`bird_verified_cot_2462.jsonl` holds the 2,462 verified questions of the
**BIRD-Verified-CoT-2462 (GPT-5.4 distilled)** dataset, reduced to the fields
the evaluation uses: `question_id`, `db_id`, `question`, `evidence`, `SQL`,
`grading_method`.

It is bundled so evaluations run inside a private network with no internet
access.

- Source: https://huggingface.co/datasets/wenyupapa/BIRD-Verified-CoT-2462-GPT5.4
  (config `valid_only`)
- Built on the BIRD benchmark (https://bird-bench.github.io), arXiv:2305.03111
- License: CC BY-SA 4.0 (https://creativecommons.org/licenses/by-sa/4.0/).
  This file is a derivative (fields removed, rows sorted by `db_id`,
  `question_id`) and is shared under the same license.

# BIRD databases (bundled subset)

`databases/` holds three BIRD train databases as SQLite files, so they can be
loaded into a private network's database with no internet access
(`python -m src.infrastructure.bird.load_bird_databases`):

| File | Questions | Size |
|---|---|---|
| `books.sqlite` | 79 | 4.3 MB |
| `car_retails.sqlite` | 56 | 0.5 MB |
| `retail_world.sqlite` | 43 | 0.1 MB |

- Source: BIRD train databases (https://bird-bench.github.io), downloaded from
  the Hugging Face mirror https://huggingface.co/datasets/premai-io/birdbench
  (`train/train_databases/<db_id>/<db_id>.sqlite`), unmodified.
  SHA-256: books `8b34bb97…`, car_retails `a5c434b6…`, retail_world `90ac33a0…`.
- License: CC BY-SA 4.0, as the BIRD benchmark.

`databases/verified_questions.json` lists, per engine, the questions whose
gold SQL still returns BIRD's answer after translation from SQLite
(`python -m src.infrastructure.bird.verify_bird_questions`). Checked on
Trino 443 (Iceberg catalog) and Postgres 16.

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

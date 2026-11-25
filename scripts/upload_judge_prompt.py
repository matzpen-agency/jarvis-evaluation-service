"""
Upload the LLM judge prompt (prompts/llm_judge_false_negative.md) to Langfuse
as a text prompt with the "production" label. Uses the service's LANGFUSE_*
settings (.env), so run it once per Langfuse (cloud, self-hosted):

    uv run python scripts/upload_judge_prompt.py [prompt name]

Each upload adds a new version; the judge reads the "production" one. Set the
evaluation service's LLM_JUDGE_PROMPT_NAME to the same name.
"""

from __future__ import annotations

import sys
from pathlib import Path

import langfuse as lf_sdk

from src.config.settings import settings as service_settings

DEFAULT_NAME = "text2sql-llm-judge-false-negative"
PROMPT_FILE = (
    Path(__file__).resolve().parent.parent / "prompts" / "llm_judge_false_negative.md"
)


def main() -> None:
    name = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_NAME
    settings = service_settings
    client = lf_sdk.Langfuse(
        public_key=settings.LANGFUSE_PUBLIC_KEY.get_secret_value(),
        secret_key=settings.LANGFUSE_SECRET_KEY.get_secret_value(),
        host=settings.LANGFUSE_HOST,
    )
    prompt = client.create_prompt(
        name=name,
        type="text",
        prompt=PROMPT_FILE.read_text(encoding="utf-8"),
        labels=["production"],
        config={"min_confidence_to_overturn": 80},
    )
    print(f"Uploaded '{name}' version {prompt.version} to {settings.LANGFUSE_HOST}")  # noqa: T201
    print(f"Variables: {', '.join(prompt.variables)}")  # noqa: T201


if __name__ == "__main__":
    main()

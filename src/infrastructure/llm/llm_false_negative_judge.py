"""
llm_false_negative_judge.py — FalseNegativeJudge backed by a chat model.

The prompt lives in Langfuse (a text prompt, label "production", uploaded by
scripts/upload_judge_prompt.py), so it can be changed without a deploy. It is
filled with the question, both results (first rows, as a table) and why the
contains check failed, and sent to any OpenAI-compatible Chat Completions
endpoint. The model replies with JSON: is_false_negative, confidence (0-100),
key_differences and reasoning.
"""

from __future__ import annotations

import asyncio
import json
import re
from typing import Any

import httpx
import structlog

from src.domain.entities.query_result import QueryResult
from src.domain.services.rate_limit import is_rate_limit, wait_before_retry
from src.ports.false_negative_judge import FalseNegativeJudge, JudgeVerdict

logger = structlog.get_logger(__name__)

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)
_MAX_ATTEMPTS = 3
# Rows of each result shown to the model; more are summarised as a count
MAX_ROWS_SHOWN = 50
_MAX_CELL_CHARS = 80


def _cell(value: Any) -> str:
    text = "NULL" if value is None else str(value)
    text = text.replace("\n", " ")
    return text if len(text) <= _MAX_CELL_CHARS else text[: _MAX_CELL_CHARS - 1] + "…"


def format_result(result: QueryResult) -> str:
    """The result as a pipe table: header, then the first rows."""
    lines = [" | ".join(_cell(c) for c in result.columns)]
    lines += [" | ".join(_cell(v) for v in row) for row in result.rows[:MAX_ROWS_SHOWN]]
    hidden = _row_count(result) - min(len(result.rows), MAX_ROWS_SHOWN)
    if hidden > 0:
        lines.append(f"... {hidden} more rows not shown")
    return "\n".join(lines)


def _row_count(result: QueryResult) -> int:
    return result.total_row_count or result.row_count or len(result.rows)


def _summary(result: QueryResult) -> dict[str, str]:
    rows, cols = _row_count(result), len(result.columns)
    key = f"columns: {', '.join(result.columns) or 'none'}"
    if result.truncated:
        key += f"; only the first {len(result.rows)} rows were returned"
    return {
        "shape": f"{rows} rows x {cols} columns",
        "row_count": str(rows),
        "key_characteristic": key,
    }


def _failure_analysis(reason: str, details: dict[str, Any]) -> str:
    """The contains check's details, minus bulky fields, as readable text."""
    shown = {
        k: v
        for k, v in details.items()
        if k != "reason" and not isinstance(v, list | dict)
    }
    text = reason.replace("_", " ")
    if shown:
        text += " (" + ", ".join(f"{k}: {v}" for k, v in shown.items()) + ")"
    return text[:1500]


def parse_verdict(content: str | None) -> dict[str, Any] | None:
    """The JSON verdict in a model reply, tolerating <think> blocks and text around it."""
    if not content:
        return None
    match = _JSON_RE.search(_THINK_RE.sub("", content))
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict) or "is_false_negative" not in data:
        return None
    value = data["is_false_negative"]
    if isinstance(value, str):
        value = value.strip().lower() in ("true", "yes", "1")
    try:
        confidence = int(float(str(data.get("confidence", "")).strip().rstrip("%")))
    except ValueError:
        confidence = None
    differences = data.get("key_differences")
    return {
        "is_false_negative": bool(value),
        "confidence": confidence,
        "reasoning": str(data.get("reasoning") or "").strip(),
        "key_differences": differences if isinstance(differences, list) else [],
    }


class LlmFalseNegativeJudge(FalseNegativeJudge):
    def __init__(
        self,
        langfuse_client: Any,
        prompt_name: str,
        base_url: str,
        api_key: str,
        model: str,
        timeout: float = 120.0,
    ) -> None:
        self._langfuse = langfuse_client
        self._prompt_name = prompt_name
        self._url = f"{base_url.rstrip('/')}/chat/completions"
        self._api_key = api_key
        self._model = model
        self._timeout = timeout

    async def judge(
        self,
        *,
        question: str,
        expected: QueryResult,
        generated: QueryResult,
        failure_reason: str,
        failure_details: dict[str, Any],
    ) -> JudgeVerdict:
        base = {"model": self._model, "prompt_name": self._prompt_name}
        try:
            prompt = await asyncio.to_thread(
                self._langfuse.get_prompt,
                self._prompt_name,
                label="production",
                type="text",
            )
        except Exception as exc:
            logger.warning("false_negative_judge.prompt_unavailable", error=str(exc))
            return JudgeVerdict(None, error=f"prompt unavailable: {exc}"[:300], **base)

        exp, gen = _summary(expected), _summary(generated)
        text = prompt.compile(
            user_question=question.strip(),
            expected_result_formatted=format_result(expected),
            expected_shape=exp["shape"],
            expected_row_count=exp["row_count"],
            expected_key_characteristic=exp["key_characteristic"],
            generated_result_formatted=format_result(generated),
            generated_shape=gen["shape"],
            generated_row_count=gen["row_count"],
            generated_key_characteristic=gen["key_characteristic"],
            automated_failure_reason=failure_reason,
            failure_analysis=_failure_analysis(failure_reason, failure_details),
        )
        base["prompt_version"] = getattr(prompt, "version", None)
        try:
            verdict = await self._ask(text)
        except Exception as exc:
            logger.warning(
                "false_negative_judge.unavailable", model=self._model, error=str(exc)
            )
            return JudgeVerdict(None, error=f"judge unavailable: {exc}"[:300], **base)
        logger.info(
            "false_negative_judge.decided",
            is_false_negative=verdict["is_false_negative"],
            confidence=verdict["confidence"],
        )
        return JudgeVerdict(**verdict, **base)

    async def _ask(self, prompt_text: str) -> dict[str, Any]:
        if "qwen" in self._model.lower():
            prompt_text += "\n/no_think"  # Qwen3: skip the long reasoning trace
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt_text}],
            "temperature": 0,
            "max_tokens": 2048,
        }
        headers = {"Authorization": f"Bearer {self._api_key}"}
        last_error: Exception | None = None
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            attempt = rate_limited = 0
            while attempt < _MAX_ATTEMPTS:
                attempt += 1
                try:
                    response = await client.post(
                        self._url, json=payload, headers=headers
                    )
                    if response.status_code == 400 and "temperature" in payload:
                        # Some models (e.g. OpenAI reasoning models) reject these
                        payload = {
                            "model": self._model,
                            "messages": payload["messages"],
                        }
                        continue
                    response.raise_for_status()
                    message = response.json()["choices"][0]["message"]
                    verdict = parse_verdict(message.get("content")) or parse_verdict(
                        message.get("reasoning_content")
                    )
                    if verdict is None:
                        raise ValueError(
                            f"unparseable reply: {str(message.get('content'))[:200]}"
                        )
                    return verdict
                except httpx.HTTPStatusError as exc:
                    last_error = exc
                    if is_rate_limit(exc):
                        # Wait out the provider's rate limit (raises when out of
                        # retries or stopped); it doesn't use up the other attempts
                        rate_limited += 1
                        await wait_before_retry("llm_judge", rate_limited, exc)
                        attempt -= 1
                        continue
                    if exc.response.status_code < 500:
                        break
                except (httpx.TransportError, ValueError, KeyError) as exc:
                    last_error = exc
                await asyncio.sleep(1.5 * attempt)
        raise RuntimeError(str(last_error) if last_error else "no reply")

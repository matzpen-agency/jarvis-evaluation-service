"""
llm_order_judge.py — OrderJudge backed by an OpenAI-compatible chat model.

Talks to any OpenAI-compatible Chat Completions endpoint over httpx:
OpenAI itself, or a private-network AI service serving e.g.
Qwen/Qwen3.6-27B-FP8, Qwen/Qwen3.8-27B-FP8 or openai/gpt-oss-120b.

Each question is judged once: decisions are cached in memory and in a JSON
file keyed by the question text, so every run treats a question the same way
and a question is only re-judged when its text changes.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
from datetime import UTC, datetime
from typing import Any

import httpx
import structlog

from src.ports.order_judge import OrderDecision, OrderJudge

logger = structlog.get_logger(__name__)

SYSTEM_PROMPT = """You review analytics questions that will be answered with a SQL query.
Decide whether the question asks for the result rows in a specific ORDER.

Order IS requested when the wording asks to sort or rank the rows: "sorted by",
"ordered by", "from highest to lowest", "in descending order", "chronologically",
"alphabetically", "rank", "top 10 ... in order", "list ... by date", or the Hebrew
equivalents (ממוין, לפי סדר, מהגבוה לנמוך, מהנמוך לגבוה, בסדר יורד/עולה, דרג).

Order is NOT requested when the question only asks which rows, how many, a
total, an average, or a breakdown per group ("revenue by territory", "sales per
year", "כמה ... בכל ..."). "By X" meaning "grouped by X" is not an order request.
If unsure, answer false.

The question may be in Hebrew or English. Reply with JSON only, no other text:
{"requires_order": true or false, "reason": "<one short sentence>"}

Examples:
Q: What is our revenue by sales territory?
A: {"requires_order": false, "reason": "Asks for a breakdown per territory, not a sort."}
Q: List the 10 customers with the highest yearly income, from highest to lowest.
A: {"requires_order": true, "reason": "Explicitly asks for highest-to-lowest order."}
Q: הצג את המוצרים לפי מחיר מהגבוה לנמוך
A: {"requires_order": true, "reason": "Asks to show products from highest to lowest price."}
Q: כמה לקוחות יש בכל מדינה?
A: {"requires_order": false, "reason": "Asks for a count per country only."}
Q: Show monthly sales for 2013 in chronological order.
A: {"requires_order": true, "reason": "Asks for chronological order."}"""

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)
_MAX_ATTEMPTS = 3


def parse_decision(content: str | None) -> tuple[bool, str] | None:
    """Extract (requires_order, reason) from a model reply, tolerating
    reasoning blocks (<think>…</think>) and text around the JSON."""
    if not content:
        return None
    text = _THINK_RE.sub("", content).strip()
    match = _JSON_RE.search(text)
    if match:
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            data = None
        if isinstance(data, dict) and "requires_order" in data:
            value = data["requires_order"]
            if isinstance(value, str):
                value = value.strip().lower() in ("true", "yes", "1")
            return bool(value), str(data.get("reason") or "").strip()
    lowered = text.lower()
    if '"requires_order": true' in lowered or "requires_order: true" in lowered:
        return True, ""
    if '"requires_order": false' in lowered or "requires_order: false" in lowered:
        return False, ""
    return None


class LlmOrderJudge(OrderJudge):
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        timeout: float = 60.0,
        cache_path: str | None = None,
    ) -> None:
        self._url = f"{base_url.rstrip('/')}/chat/completions"
        self._api_key = api_key
        self._model = model
        self._timeout = timeout
        self._cache_path = cache_path
        self._cache: dict[str, dict[str, Any]] | None = None
        self._lock = asyncio.Lock()

    async def requires_order(self, question: str) -> OrderDecision:
        key = hashlib.sha256(question.strip().encode("utf-8")).hexdigest()
        async with self._lock:
            cached = self._load_cache().get(key)
            if cached is not None:
                return OrderDecision(cached["requires_order"], cached["reason"], "cache")

            try:
                requires, reason = await self._ask(question)
            except Exception as exc:
                logger.warning("order_judge.unavailable", model=self._model, error=str(exc))
                return OrderDecision(None, f"order judge unavailable: {exc}"[:300], "unavailable")

            self._cache[key] = {
                "requires_order": requires,
                "reason": reason,
                "model": self._model,
                "question": question.strip()[:300],
                "decided_at": datetime.now(tz=UTC).isoformat(),
            }
            self._save_cache()
            logger.info("order_judge.decided", requires_order=requires, model=self._model)
            return OrderDecision(requires, reason, "llm")

    async def _ask(self, question: str) -> tuple[bool, str]:
        user = f"Q: {question.strip()}\nA:"
        if "qwen" in self._model.lower():
            user += " /no_think"  # Qwen3: skip the long reasoning trace
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user},
            ],
            "temperature": 0,
            "max_tokens": 1024,
        }
        headers = {"Authorization": f"Bearer {self._api_key}"}

        last_error: Exception | None = None
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            for attempt in range(1, _MAX_ATTEMPTS + 1):
                try:
                    response = await client.post(self._url, json=payload, headers=headers)
                    if response.status_code == 400 and "temperature" in payload:
                        # Some models (e.g. OpenAI reasoning models) reject these
                        payload = {"model": self._model, "messages": payload["messages"]}
                        continue
                    response.raise_for_status()
                    message = response.json()["choices"][0]["message"]
                    decision = parse_decision(message.get("content")) or parse_decision(
                        message.get("reasoning_content")
                    )
                    if decision is None:
                        raise ValueError(f"unparseable reply: {str(message.get('content'))[:200]}")
                    return decision
                except httpx.HTTPStatusError as exc:
                    last_error = exc
                    if exc.response.status_code < 500 and exc.response.status_code != 429:
                        break
                except (httpx.TransportError, ValueError, KeyError) as exc:
                    last_error = exc
                await asyncio.sleep(1.5 * attempt)
        raise RuntimeError(str(last_error) if last_error else "no reply")

    def _load_cache(self) -> dict[str, dict[str, Any]]:
        if self._cache is None:
            self._cache = {}
            if self._cache_path and os.path.exists(self._cache_path):
                try:
                    with open(self._cache_path, encoding="utf-8") as f:
                        self._cache = json.load(f)
                except (OSError, json.JSONDecodeError) as exc:
                    logger.warning("order_judge.cache_unreadable", error=str(exc))
        return self._cache

    def _save_cache(self) -> None:
        if not self._cache_path:
            return
        try:
            os.makedirs(os.path.dirname(self._cache_path) or ".", exist_ok=True)
            tmp = f"{self._cache_path}.tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self._cache, f, ensure_ascii=False, indent=1)
            os.replace(tmp, self._cache_path)
        except OSError as exc:
            logger.warning("order_judge.cache_write_failed", error=str(exc))

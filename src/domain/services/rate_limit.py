"""
rate_limit.py — Waiting out the LLM provider's rate limit (HTTP 429).

Every place that calls an LLM — the agent writing the SQL, the LLM judge, the
order judge — retries a rate-limited call up to LLM_RATE_LIMIT_MAX_RETRIES
times, waiting LLM_RATE_LIMIT_RETRY_WAIT_SECONDS before the first retry and
twice as long before each next one (20s, 40s, 80s by default).

While a question waits, a run's RateLimitTracker lists it (so the studio can
show it) and the user can stop the wait: the call then fails at once and the
question is marked rate limited, to be rerun later.

The question being evaluated travels in a context variable, so the judges —
which don't know which question they judge — report under the right one.
"""

from __future__ import annotations

import asyncio
import contextvars
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

# The error of a question that the rate limit stopped (the studio offers a rerun)
RATE_LIMITED = "rate_limited"

_RATE_LIMIT_TEXT = re.compile(
    r"\b429\b|rate[ _-]?limit|too many requests|ratelimiterror|quota exceeded|"
    r"resource[ _]exhausted",
    re.IGNORECASE,
)


def is_rate_limit(error: object) -> bool:
    """True when an exception, HTTP status or message says the LLM provider rate limited us."""
    if error is None:
        return False
    if isinstance(error, int):
        return error == 429
    response = getattr(error, "response", None)
    if getattr(response, "status_code", None) == 429:
        return True
    return bool(_RATE_LIMIT_TEXT.search(str(error)))


@dataclass(frozen=True)
class RateLimitPolicy:
    max_retries: int = 3
    wait_seconds: float = 20.0

    def delay(self, retry: int) -> float:
        """Seconds to wait before retry number `retry` (1-based): doubling each time."""
        return self.wait_seconds * (2 ** max(retry - 1, 0))


class RateLimitStoppedError(Exception):
    """The user stopped this question's rate-limit retries."""


class RateLimitExhaustedError(Exception):
    """The call was still rate limited after every retry."""


@dataclass
class _Waiting:
    item_id: str
    question: str
    stage: str
    retry: int
    max_retries: int
    retry_at: datetime
    error: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "question_id": self.item_id,
            "question": self.question,
            "stage": self.stage,
            "retry": self.retry,
            "max_retries": self.max_retries,
            "retry_at": self.retry_at.isoformat(),
            "error": self.error,
        }


class RateLimitTracker:
    """The questions of one run that wait out a rate limit, and their stop switches."""

    def __init__(self) -> None:
        self._waiting: dict[str, _Waiting] = {}
        self._stops: dict[str, asyncio.Event] = {}
        self.retried: dict[str, int] = {}  # question id -> retries so far

    def stop_event(self, item_id: str) -> asyncio.Event:
        return self._stops.setdefault(item_id, asyncio.Event())

    def stop(self, item_id: str) -> bool:
        """Stop the question's retries. False when it isn't waiting or retrying."""
        if item_id not in self._waiting and item_id not in self.retried:
            return False
        self.stop_event(item_id).set()
        return True

    def stopped(self, item_id: str) -> bool:
        event = self._stops.get(item_id)
        return bool(event and event.is_set())

    def waiting(self) -> list[dict[str, Any]]:
        return [w.as_dict() for w in self._waiting.values()]

    def _start_wait(self, waiting: _Waiting) -> None:
        self._waiting[waiting.item_id] = waiting
        self.retried[waiting.item_id] = self.retried.get(waiting.item_id, 0) + 1

    def _end_wait(self, item_id: str) -> None:
        self._waiting.pop(item_id, None)


@dataclass
class QuestionScope:
    """The question an LLM call belongs to, and what the rate limit did to it."""

    item_id: str
    question: str
    policy: RateLimitPolicy
    tracker: RateLimitTracker | None = None
    # Set when a call gave up: (stage, "exhausted" | "stopped", last error)
    gave_up: tuple[str, str, str] | None = None
    retries: dict[str, int] = field(default_factory=dict)

    def error_label(self) -> str | None:
        if self.gave_up is None:
            return None
        stage, why, error = self.gave_up
        if why == "stopped":
            return f"{RATE_LIMITED}: {stage} (retries stopped by the user)"
        return f"{RATE_LIMITED}: {stage} (still rate limited after {self.policy.max_retries} retries: {error[:200]})"


_current: contextvars.ContextVar[QuestionScope | None] = contextvars.ContextVar(
    "rate_limit_question", default=None
)
_default_policy = RateLimitPolicy()


def configure(policy: RateLimitPolicy) -> None:
    """The policy for LLM calls made outside a question (set once from settings)."""
    global _default_policy
    _default_policy = policy


def enter_question(scope: QuestionScope) -> contextvars.Token:
    return _current.set(scope)


def leave_question(token: contextvars.Token) -> None:
    _current.reset(token)


def current_question() -> QuestionScope | None:
    return _current.get()


async def wait_before_retry(stage: str, retry: int, error: object) -> None:
    """
    Wait before retry number `retry` of a rate-limited call (1-based).

    Raises RateLimitExhaustedError when there are no retries left, and
    RateLimitStoppedError when the user stops the question's retries (also during
    the wait). Both are recorded on the question, which is then marked rate
    limited.
    """
    scope = _current.get()
    policy = scope.policy if scope else _default_policy
    text = str(error)[:300]
    tracker = scope.tracker if scope else None
    if scope and tracker and tracker.stopped(scope.item_id):
        scope.gave_up = (stage, "stopped", text)
        raise RateLimitStoppedError(stage)
    if retry > policy.max_retries:
        if scope:
            scope.gave_up = (stage, "exhausted", text)
        raise RateLimitExhaustedError(text)

    delay = policy.delay(retry)
    logger.warning(
        "rate_limit.waiting",
        stage=stage,
        retry=retry,
        max_retries=policy.max_retries,
        wait_seconds=delay,
        question_id=scope.item_id if scope else None,
        error=text,
    )
    if scope is None or tracker is None:
        await asyncio.sleep(delay)
        return

    scope.retries[stage] = scope.retries.get(stage, 0) + 1
    tracker._start_wait(
        _Waiting(
            item_id=scope.item_id,
            question=scope.question,
            stage=stage,
            retry=retry,
            max_retries=policy.max_retries,
            retry_at=datetime.now(tz=UTC) + timedelta(seconds=delay),
            error=text,
        )
    )
    stop = tracker.stop_event(scope.item_id)
    try:
        await asyncio.wait_for(stop.wait(), timeout=delay)
    except TimeoutError:
        return  # waited the full delay: retry
    finally:
        tracker._end_wait(scope.item_id)
    scope.gave_up = (stage, "stopped", text)
    logger.info("rate_limit.stopped_by_user", question_id=scope.item_id, stage=stage)
    raise RateLimitStoppedError(stage)

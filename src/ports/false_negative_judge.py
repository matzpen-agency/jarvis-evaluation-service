"""
false_negative_judge.py — A second look at questions the metrics failed.

The contains check compares results strictly, so a correct answer can fail
on order, formatting, NULL vs 0 or an extra total row. The judge reads the
question and both results and decides whether the generated result answers
the question anyway (a false negative).

Implementations: LlmFalseNegativeJudge
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from src.domain.entities.query_result import QueryResult


@dataclass(frozen=True)
class JudgeVerdict:
    """is_false_negative is None when no verdict could be reached (error set)."""

    is_false_negative: bool | None
    confidence: int | None = None
    reasoning: str = ""
    key_differences: list[dict[str, Any]] = field(default_factory=list)
    model: str | None = None
    prompt_name: str | None = None
    prompt_version: int | None = None
    error: str | None = None


class FalseNegativeJudge(ABC):
    @abstractmethod
    async def judge(
        self,
        *,
        question: str,
        expected: QueryResult,
        generated: QueryResult,
        failure_reason: str,
        failure_details: dict[str, Any],
    ) -> JudgeVerdict:
        """Does `generated` answer `question` correctly although the metrics failed it?"""
        ...

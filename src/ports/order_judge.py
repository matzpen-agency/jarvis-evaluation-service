"""
order_judge.py — Decides whether a natural-language question asks for its
results in a specific order.

Used only for the one case the SQL can't settle: the agent returned exactly
the golden rows, in a different order, and the golden SQL has ORDER BY
without LIMIT (so the order is either requested or incidental).

Implementations: LlmOrderJudge
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class OrderDecision:
    """requires_order is None when no decision could be made (e.g. LLM down)."""

    requires_order: bool | None
    reason: str
    source: str  # "llm" | "cache" | "unavailable"


class OrderJudge(ABC):
    @abstractmethod
    async def requires_order(self, question: str) -> OrderDecision:
        """Does `question` (Hebrew or English) ask for a specific row order?"""
        ...

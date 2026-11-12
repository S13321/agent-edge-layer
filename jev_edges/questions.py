"""Typed edge questions (Choice / Score / Noul) and their normalized answers."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Union


@dataclass(frozen=True)
class Choice:
    """Pick exactly one label from a closed set."""

    labels: tuple[str, ...]
    prompt: str = "Which option fits the current graph state?"

    def __post_init__(self) -> None:
        if not self.labels or len(set(self.labels)) != len(self.labels):
            raise ValueError("Choice needs at least one label and no duplicates")


@dataclass(frozen=True)
class Score:
    """Grade on a 1..N rubric; rubric[i] describes level i+1."""

    rubric: tuple[str, ...]
    prompt: str = "Score the current graph state on this rubric."

    def __post_init__(self) -> None:
        if len(self.rubric) < 2:
            raise ValueError("Score needs a rubric of at least two levels")


@dataclass(frozen=True)
class Noul:
    """Yes/no question read as a probability of 'yes'."""

    prompt: str


Question = Union[Choice, Score, Noul]

YES, NO = "yes", "no"


def outcomes(q: Question) -> tuple[str, ...]:
    if isinstance(q, Choice):
        return q.labels
    if isinstance(q, Score):
        return tuple(str(i) for i in range(1, len(q.rubric) + 1))
    return (YES, NO)


class InvalidDistribution(ValueError):
    pass


def normalize(q: Question, raw: Mapping[str, float]) -> dict[str, float]:
    """Validate a backend distribution against the question's outcome set.

    Unknown outcomes or negative mass are rejected rather than silently dropped,
    so a misbehaving backend is treated as unsure and escalated.
    """
    allowed = outcomes(q)
    extra = set(raw) - set(allowed)
    if extra:
        raise InvalidDistribution(f"unknown outcomes {sorted(extra)}")
    probs = {o: float(raw.get(o, 0.0)) for o in allowed}
    if any(p < 0 for p in probs.values()):
        raise InvalidDistribution("negative probability")
    total = sum(probs.values())
    if total <= 0:
        raise InvalidDistribution("empty distribution")
    return {o: p / total for o, p in probs.items()}


@dataclass(frozen=True)
class Answer:
    key: str
    question: Question
    probs: Mapping[str, float]
    tier: str
    resolved: bool = True
    notes: tuple[str, ...] = field(default_factory=tuple)
    # (tier, normalized probs) for every tier that gave a valid answer, cheapest first.
    history: tuple[tuple[str, Mapping[str, float]], ...] = field(default_factory=tuple)

    @property
    def top(self) -> str:
        # Ties break by declared outcome order, never by dict/hash order.
        order = outcomes(self.question)
        return max(order, key=lambda o: (self.probs.get(o, 0.0), -order.index(o)))

    @property
    def confidence(self) -> float:
        return self.probs.get(self.top, 0.0)

    @property
    def p_yes(self) -> float:
        if not isinstance(self.question, Noul):
            raise TypeError("p_yes is only defined for Noul questions")
        return self.probs.get(YES, 0.0)

    @property
    def level(self) -> int:
        if not isinstance(self.question, Score):
            raise TypeError("level is only defined for Score questions")
        return int(self.top)

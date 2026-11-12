"""Calibrated-probability tiered routing: ask the cheap tier, escalate only unsure questions."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

from .backends import DecisionBackend
from .questions import Answer, InvalidDistribution, Noul, Question, YES, normalize


@dataclass(frozen=True)
class Threshold:
    """When a tier's answer counts as settled.

    Choice/Score: top probability >= ``min_confidence``.
    Noul: p(yes) >= ``yes_at`` (settled yes) or p(yes) <= ``no_at`` (settled no);
    anything in between is unsure and goes up a tier.
    """

    min_confidence: float = 0.7
    yes_at: float = 0.8
    no_at: float = 0.2

    def __post_init__(self) -> None:
        if not 0.0 <= self.no_at < self.yes_at <= 1.0:
            raise ValueError("need 0 <= no_at < yes_at <= 1")

    def settled(self, ans: Answer) -> bool:
        if isinstance(ans.question, Noul):
            p = ans.probs.get(YES, 0.0)
            return p >= self.yes_at or p <= self.no_at
        return ans.confidence >= self.min_confidence


@dataclass(frozen=True)
class TierCall:
    tier: str
    keys: tuple[str, ...]
    cost_usd: float
    error: Optional[str] = None


@dataclass(frozen=True)
class Decision:
    answers: Mapping[str, Answer]
    trace: tuple[TierCall, ...]

    def __getitem__(self, key: str) -> Answer:
        return self.answers[key]

    def __contains__(self, key: object) -> bool:
        return key in self.answers

    @property
    def cost_usd(self) -> float:
        return sum(c.cost_usd for c in self.trace)

    @property
    def escalated(self) -> tuple[str, ...]:
        return tuple(k for k, a in self.answers.items() if a.tier != self.trace[0].tier) if self.trace else ()


class TieredDecider:
    """Runs all questions as ONE request per tier; only unsettled keys reach the next tier.

    A key still unsettled after the last tier comes back with ``resolved=False`` so the
    edge policy can apply its conservative default instead of trusting a guess.
    """

    def __init__(
        self,
        tiers: Sequence[DecisionBackend],
        thresholds: Optional[Mapping[str, Threshold]] = None,
        default: Threshold = Threshold(),
    ) -> None:
        if not tiers:
            raise ValueError("at least one tier is required")
        self.tiers = tuple(tiers)
        self.thresholds = dict(thresholds or {})
        self.default = default

    def threshold_for(self, key: str) -> Threshold:
        # "safe.payment" falls back to the "safe" family threshold.
        return self.thresholds.get(key) or self.thresholds.get(key.split(".", 1)[0]) or self.default

    def decide(self, state: Mapping[str, Any], questions: Mapping[str, Question]) -> Decision:
        pending = dict(questions)
        settled: dict[str, Answer] = {}
        last_seen: dict[str, Answer] = {}
        notes: dict[str, list[str]] = {k: [] for k in questions}
        history: dict[str, list[tuple[str, Mapping[str, float]]]] = {k: [] for k in questions}
        trace: list[TierCall] = []

        for backend in self.tiers:
            if not pending:
                break
            try:
                result = backend.decide(state, pending)
            except Exception as exc:  # network, refusal, malformed reply: escalate, don't crash the hop
                trace.append(TierCall(backend.name, tuple(pending), 0.0, f"{type(exc).__name__}: {exc}"))
                for k in pending:
                    notes[k].append(f"{backend.name}: backend error")
                continue
            trace.append(TierCall(backend.name, tuple(pending), result.cost_usd))

            unsure: dict[str, Question] = {}
            for key, q in pending.items():
                try:
                    probs = normalize(q, result.distributions[key])
                except (KeyError, TypeError, ValueError, InvalidDistribution) as exc:
                    notes[key].append(f"{backend.name}: invalid answer ({exc})")
                    unsure[key] = q
                    continue
                ans = Answer(key, q, probs, backend.name)
                history[key].append((backend.name, probs))
                if self.threshold_for(key).settled(ans):
                    settled[key] = ans
                else:
                    notes[key].append(f"{backend.name}: unsure ({ans.top} @ {ans.confidence:.2f})")
                    last_seen[key] = ans
                    unsure[key] = q
            pending = unsure

        for key, q in pending.items():
            prev = last_seen.get(key)
            settled[key] = Answer(
                key, q, dict(prev.probs) if prev else {}, prev.tier if prev else "none", resolved=False
            )

        answers = {
            k: Answer(a.key, a.question, a.probs, a.tier, a.resolved, tuple(notes[k]), tuple(history[k]))
            for k in questions
            for a in (settled[k],)
        }
        return Decision(answers, tuple(trace))

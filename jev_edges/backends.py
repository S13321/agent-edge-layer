"""Decision backends: the fast Jev tier and higher escalation tiers."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional, Protocol

from .questions import Choice, Noul, Question, Score, outcomes


@dataclass(frozen=True)
class DecideResult:
    distributions: Mapping[str, Mapping[str, float]]
    cost_usd: float = 0.0


class DecisionBackend(Protocol):
    name: str

    def decide(self, state: Mapping[str, Any], questions: Mapping[str, Question]) -> DecideResult:
        """Answer every question in ONE request; return a raw distribution per key."""


class BackendError(RuntimeError):
    pass


# --------------------------------------------------------------------------- Jev


def _default_read_probs(response: Any, key: str) -> Mapping[str, float]:
    item = response[key] if isinstance(response, Mapping) else getattr(response, key)
    return item["probs"] if isinstance(item, Mapping) else item.probs


def _default_input_tokens(response: Any) -> Optional[int]:
    usage = response.get("usage") if isinstance(response, Mapping) else getattr(response, "usage", None)
    if usage is None:
        return None
    return usage["input_tokens"] if isinstance(usage, Mapping) else getattr(usage, "input_tokens", None)


class JevBackend:
    """Adapter over a TypeSafe Jev client exposing ``decide(model=, state=, questions=)``.

    The working note's snippet is a sketch ("exact fields in TypeSafe docs"), so the
    SDK-specific seams are injectable: ``to_sdk`` maps our question objects to the
    SDK's own Choice/Score/Noul, ``read_probs`` pulls one question's distribution off
    the response, and ``input_tokens_of`` reads usage for cost accounting.
    """

    PRICE_PER_M_INPUT = 0.042  # USD per 1M input tokens, output free (TypeSafe list price)

    def __init__(
        self,
        client: Any,
        model: str = "jev-latest",
        *,
        name: str = "jev",
        to_sdk: Callable[[Question], Any] = lambda q: q,
        read_probs: Callable[[Any, str], Mapping[str, float]] = _default_read_probs,
        input_tokens_of: Callable[[Any], Optional[int]] = _default_input_tokens,
    ) -> None:
        self.client = client
        self.model = model
        self.name = name
        self._to_sdk = to_sdk
        self._read_probs = read_probs
        self._input_tokens_of = input_tokens_of

    def decide(self, state: Mapping[str, Any], questions: Mapping[str, Question]) -> DecideResult:
        response = self.client.decide(
            model=self.model,
            state=state,
            questions={k: self._to_sdk(q) for k, q in questions.items()},
        )
        dists = {k: self._read_probs(response, k) for k in questions}
        tokens = self._input_tokens_of(response)
        cost = tokens * self.PRICE_PER_M_INPUT / 1_000_000 if tokens is not None else 0.0
        return DecideResult(dists, cost)


# ------------------------------------------------------------------------ Claude

_SYSTEM = (
    "You are the edge router of a multi-agent graph. For each question you receive, "
    "return a probability distribution over exactly the listed outcomes. The numbers must "
    "reflect your real uncertainty given the graph state: put mass on more than one outcome "
    "when the state does not settle the question. Do not explain."
)

# Models that accept the server-side `fallbacks: "default"` form.
_FALLBACK_MODELS = {"claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5", "claude-fable-5-2", "claude-fable-5-1"}


def _describe(key: str, q: Question) -> str:
    if isinstance(q, Choice):
        return f"- {key} (choice): {q.prompt} Outcomes: {', '.join(q.labels)}."
    if isinstance(q, Score):
        levels = "; ".join(f"{i} = {d}" for i, d in enumerate(q.rubric, start=1))
        return f"- {key} (score): {q.prompt} Rubric: {levels}."
    return f"- {key} (yes/no): {q.prompt} Outcomes: yes, no."


def build_schema(questions: Mapping[str, Question]) -> dict:
    props = {}
    for key, q in questions.items():
        outs = outcomes(q)
        props[key] = {
            "type": "object",
            "properties": {o: {"type": "number"} for o in outs},
            "required": list(outs),
            "additionalProperties": False,
        }
    return {
        "type": "object",
        "properties": props,
        "required": list(questions),
        "additionalProperties": False,
    }


def render_prompt(state: Mapping[str, Any], questions: Mapping[str, Question]) -> str:
    # sort_keys keeps identical states byte-identical (deterministic + cache friendly).
    return (
        "<graph_state>\n"
        + json.dumps(state, sort_keys=True, ensure_ascii=False, default=str)
        + "\n</graph_state>\n\nQuestions:\n"
        + "\n".join(_describe(k, q) for k, q in questions.items())
    )


class ClaudeBackend:
    """Escalation tier on the Claude API: same typed questions, one structured-output call."""

    def __init__(
        self,
        client: Any,
        model: str = "claude-opus-5-5",
        *,
        name: Optional[str] = None,
        effort: str = "low",
        input_price_per_m: float = 4.0,
        output_price_per_m: float = 20.0,
        max_tokens: int = 8000,
    ) -> None:
        self.client = client
        self.model = model
        self.name = name or model
        self.effort = effort
        self.input_price_per_m = input_price_per_m
        self.output_price_per_m = output_price_per_m
        self.max_tokens = max_tokens

    def decide(self, state: Mapping[str, Any], questions: Mapping[str, Question]) -> DecideResult:
        kwargs: dict[str, Any] = dict(
            model=self.model,
            max_tokens=self.max_tokens,
            system=_SYSTEM,
            messages=[{"role": "user", "content": render_prompt(state, questions)}],
            output_config={
                "effort": self.effort,
                "format": {"type": "json_schema", "schema": build_schema(questions)},
            },
        )
        if self.model in _FALLBACK_MODELS:
            kwargs["betas"] = ["server-side-fallback-2026-07-01"]
            kwargs["fallbacks"] = "default"
        response = self.client.beta.messages.create(**kwargs)

        if response.stop_reason == "refusal":
            raise BackendError(f"{self.name} refused the edge decision")
        if response.stop_reason == "max_tokens":
            raise BackendError(f"{self.name} hit max_tokens before answering")
        text = next((b.text for b in response.content if b.type == "text"), None)
        if text is None:
            raise BackendError(f"{self.name} returned no text block")

        usage = response.usage
        cost = (
            usage.input_tokens * self.input_price_per_m + usage.output_tokens * self.output_price_per_m
        ) / 1_000_000
        return DecideResult(json.loads(text), cost)

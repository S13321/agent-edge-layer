"""Deterministic edge decision layer: four typed questions per hop -> one HopPlan."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Optional

from .questions import YES, Choice, Noul, Question, Score
from .router import Decision, Threshold, TieredDecider
from .safety import RISK_RULES, SAFE_THRESHOLD, check_safe_thresholds

DEFAULT_AGENTS = ("research", "code", "write", "ask")

DEFAULT_EFFORT_RUBRIC = (
    "one pass: rename, typo, single lookup",
    "light: small edit or short answer with one source",
    "standard: multi-step task inside one component",
    "heavy: cross-component change or synthesis of many sources",
    "deep plan: redesign or open-ended investigation",
)

DEFAULT_RISK_RULES = RISK_RULES

DEFAULT_THRESHOLDS: Mapping[str, Threshold] = {
    "agent": Threshold(min_confidence=0.70),
    "effort": Threshold(min_confidence=0.60),
    "safe": SAFE_THRESHOLD,
    "pass": Threshold(yes_at=0.80, no_at=0.30),
}


@dataclass(frozen=True)
class WorkerConfig:
    model: str
    effort: str


DEFAULT_WORKERS: Mapping[int, WorkerConfig] = {
    1: WorkerConfig("claude-haiku-5-5", "low"),
    2: WorkerConfig("claude-sonnet-5-5", "low"),
    3: WorkerConfig("claude-sonnet-5-5", "medium"),
    4: WorkerConfig("claude-opus-5-5", "high"),
    5: WorkerConfig("claude-opus-5-5", "xhigh"),
}


@dataclass(frozen=True)
class ToolCall:
    name: str
    args: Mapping[str, Any] = field(default_factory=dict)
    risks: Optional[tuple[str, ...]] = None


@dataclass(frozen=True)
class NodeResult:
    node: str
    output: str
    acceptance_check: str
    attempt: int = 1


@dataclass(frozen=True)
class HopInput:
    graph: Mapping[str, Any]
    finished: Optional[NodeResult] = None
    pending_tool: Optional[ToolCall] = None


class Action(str, Enum):
    PROCEED = "proceed"
    AWAIT_APPROVAL = "await_approval"
    RERUN = "rerun"
    HALT_ASK = "halt_ask"


@dataclass(frozen=True)
class HopPlan:
    action: Action
    next_agent: str
    effort_level: int
    worker: WorkerConfig
    merge: bool
    blocked_risks: tuple[str, ...]
    reasons: tuple[str, ...]
    decision: Decision

    @property
    def cost_usd(self) -> float:
        return self.decision.cost_usd


def classify_risks(tool: ToolCall, rules: Mapping[str, tuple[str, ...]] = DEFAULT_RISK_RULES) -> tuple[str, ...]:
    if tool.risks is not None:
        return tuple(dict.fromkeys(tool.risks))
    name = tool.name.lower()
    return tuple(risk for risk, needles in rules.items() if any(n in name for n in needles))


class EdgeDecisionLayer:
    """Replaces the planner-model prompt on an edge with one typed, tiered decision call."""

    def __init__(
        self,
        decider: TieredDecider,
        *,
        agents: tuple[str, ...] = DEFAULT_AGENTS,
        ask_label: str = "ask",
        effort_rubric: tuple[str, ...] = DEFAULT_EFFORT_RUBRIC,
        workers: Mapping[int, WorkerConfig] = DEFAULT_WORKERS,
        risk_rules: Mapping[str, tuple[str, ...]] = DEFAULT_RISK_RULES,
        max_reruns: int = 2,
    ) -> None:
        if ask_label not in agents:
            raise ValueError("ask_label must be one of the agents")
        missing = set(range(1, len(effort_rubric) + 1)) - set(workers)
        if missing:
            raise ValueError(f"no worker configured for effort levels {sorted(missing)}")
        check_safe_thresholds(decider.thresholds, decider.default)
        self.decider = decider
        self.agents = agents
        self.ask_label = ask_label
        self.effort_rubric = effort_rubric
        self.workers = workers
        self.risk_rules = risk_rules
        self.max_reruns = max_reruns

    # -- question assembly ---------------------------------------------------

    def questions_for(self, hop: HopInput) -> dict[str, Question]:
        qs: dict[str, Question] = {
            "agent": Choice(self.agents, "Which agent should take the next step?"),
            "effort": Score(self.effort_rubric, "How much effort does the next step need?"),
        }
        if hop.pending_tool is not None:
            for risk in classify_risks(hop.pending_tool, self.risk_rules):
                qs[f"safe.{risk}"] = Noul(
                    f"The next step calls `{hop.pending_tool.name}` ({risk}). "
                    "Can this call be undone, or is it clearly safe to run without human review?"
                )
        if hop.finished is not None:
            qs["pass"] = Noul(
                f"Does the output of node `{hop.finished.node}` meet its acceptance check: "
                f"{hop.finished.acceptance_check}"
            )
        return qs

    @staticmethod
    def state_for(hop: HopInput) -> dict[str, Any]:
        state: dict[str, Any] = {"graph": dict(hop.graph)}
        if hop.finished is not None:
            state["finished_node"] = {
                "node": hop.finished.node,
                "output": hop.finished.output,
                "acceptance_check": hop.finished.acceptance_check,
                "attempt": hop.finished.attempt,
            }
        if hop.pending_tool is not None:
            state["pending_tool_call"] = {"name": hop.pending_tool.name, "args": dict(hop.pending_tool.args)}
        return state

    # -- decide + plan -------------------------------------------------------

    def decide(self, hop: HopInput) -> HopPlan:
        decision = self.decider.decide(self.state_for(hop), self.questions_for(hop))
        return self.plan(hop, decision)

    def _effort(self, decision: Decision, reasons: list[str]) -> int:
        ans = decision["effort"]
        top = len(self.effort_rubric)
        if ans.resolved:
            return ans.level
        if not ans.probs:
            reasons.append(f"effort unresolved with no estimate -> max level {top}")
            return top
        # Unsure: round the expected level UP so an under-budgeted worker is the rarer failure.
        expected = sum(int(lvl) * p for lvl, p in ans.probs.items())
        level = min(top, max(1, math.ceil(expected - 1e-9)))
        reasons.append(f"effort unresolved -> ceil(expected {expected:.2f}) = {level}")
        return level

    def plan(self, hop: HopInput, decision: Decision) -> HopPlan:
        """Pure function of (hop, decision): same probabilities always give the same plan."""
        reasons: list[str] = []
        effort = self._effort(decision, reasons)
        worker = self.workers[effort]

        def make(action: Action, agent: str, merge: bool, blocked: tuple[str, ...] = ()) -> HopPlan:
            return HopPlan(action, agent, effort, worker, merge, blocked, tuple(reasons), decision)

        # 1. Good enough? A miss reruns the node, then reaches a human.
        merge = False
        if "pass" in decision:
            ans = decision["pass"]
            accepted = ans.resolved and ans.top == YES
            if not accepted:
                why = "unresolved" if not ans.resolved else f"p(pass)={ans.p_yes:.2f}"
                assert hop.finished is not None
                if hop.finished.attempt <= self.max_reruns:
                    reasons.append(f"output rejected ({why}) -> rerun attempt {hop.finished.attempt + 1}")
                    return make(Action.RERUN, hop.finished.node, False)
                reasons.append(f"output rejected ({why}) after {hop.finished.attempt} attempts -> ask human")
                return make(Action.HALT_ASK, self.ask_label, False)
            merge = True

        # 2. Which agent? Unsure routing goes to the human-ask agent.
        agent_ans = decision["agent"]
        if agent_ans.resolved:
            agent = agent_ans.top
        else:
            agent = self.ask_label
            reasons.append("agent unresolved on every tier -> ask")
        if agent == self.ask_label:
            return make(Action.HALT_ASK, agent, merge)

        # 3. Safe to act? Every risk must be a settled yes.
        blocked = tuple(
            k.split(".", 1)[1]
            for k, a in decision.answers.items()
            if k.startswith("safe.") and not (a.resolved and a.top == YES)
        )
        if blocked:
            reasons.append(f"irreversible action not cleared: {', '.join(blocked)}")
            return make(Action.AWAIT_APPROVAL, agent, merge, blocked)

        return make(Action.PROCEED, agent, merge)

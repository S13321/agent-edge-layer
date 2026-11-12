"""Simulated hops through the edge layer with stand-in backends (no network).

Swap the fakes for JevBackend(typesafe_client) and ClaudeBackend(anthropic.Anthropic())
to run it for real.
"""
from jev_edges import (
    DEFAULT_THRESHOLDS, DecideResult, EdgeDecisionLayer, HopInput, NodeResult, TieredDecider, ToolCall,
)

OPUS_PER_EDGE = 0.1761  # TypeSafe benchmark, Sep 2026: planner-model cost per edge decision


class Scripted:
    """Returns canned distributions; stands in for a real backend."""

    def __init__(self, name, cost, script):
        self.name, self.cost, self.script = name, cost, script

    def decide(self, state, questions):
        goal = state["graph"]["goal"]
        return DecideResult({k: v for k, v in self.script[goal].items() if k in questions}, self.cost)


jev = Scripted("jev", 0.0004, {
    "rename var": {"agent": {"code": 0.93, "research": 0.07}, "effort": {"1": 0.9, "2": 0.1}},
    "redesign auth": {"agent": {"code": 0.48, "research": 0.42, "write": 0.1}, "effort": {"5": 0.8, "4": 0.2}},
    "purge old users": {"agent": {"code": 0.85, "ask": 0.15}, "effort": {"2": 0.7, "3": 0.3},
                        "safe.delete": {"yes": 0.04, "no": 0.96}},
    "draft release notes": {"agent": {"write": 0.9, "research": 0.1}, "effort": {"2": 0.8, "3": 0.2},
                            "pass": {"yes": 0.25, "no": 0.75}},
})
opus = Scripted("claude-opus-5-5", OPUS_PER_EDGE, {
    "redesign auth": {"agent": {"research": 0.86, "code": 0.14}},
})

layer = EdgeDecisionLayer(TieredDecider([jev, opus], thresholds=DEFAULT_THRESHOLDS))
hops = [
    HopInput(graph={"goal": "rename var"}),
    HopInput(graph={"goal": "redesign auth"}),
    HopInput(graph={"goal": "purge old users"}, pending_tool=ToolCall("delete_users", {"older_than": "2y"})),
    HopInput(graph={"goal": "draft release notes"},
             finished=NodeResult("notes", "v2 adds...", "lists every merged PR", attempt=1)),
]

total = 0.0
for hop in hops:
    p = layer.decide(hop)
    total += p.cost_usd
    tiers = ", ".join(f"{k}<-{a.tier}" for k, a in p.decision.answers.items())
    print(f"{hop.graph['goal']:<20} {p.action.value:<15} agent={p.next_agent:<11} "
          f"effort={p.effort_level} worker={p.worker.model}/{p.worker.effort:<7} [{tiers}]")
    for r in p.reasons:
        print(f"{'':<20} - {r}")

print(f"\nedge cost: ${total:.4f}  vs planner on every edge: ${OPUS_PER_EDGE * len(hops):.4f}")

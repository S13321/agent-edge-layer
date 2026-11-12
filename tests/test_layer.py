import json
from types import SimpleNamespace

import pytest

from jev_edges import (
    DEFAULT_THRESHOLDS,
    Action,
    BackendError,
    ClaudeBackend,
    DecideResult,
    EdgeDecisionLayer,
    HopInput,
    JevBackend,
    NodeResult,
    Threshold,
    TieredDecider,
    ToolCall,
    classify_risks,
)
from jev_edges.questions import Choice, InvalidDistribution, Noul, normalize


class FakeBackend:
    def __init__(self, name, answers=None, cost=0.0, error=None):
        self.name = name
        self.answers = answers or {}
        self.cost = cost
        self.error = error
        self.calls = []

    def decide(self, state, questions):
        self.calls.append(tuple(questions))
        if self.error:
            raise self.error
        return DecideResult({k: self.answers[k] for k in questions if k in self.answers}, self.cost)


AGENT_CODE = {"code": 0.9, "research": 0.05, "write": 0.03, "ask": 0.02}
EFFORT_2 = {"1": 0.1, "2": 0.8, "3": 0.1}
YES_SURE = {"yes": 0.97, "no": 0.03}


def make_layer(jev, upper=None, **kw):
    tiers = [jev] + ([upper] if upper else [])
    return EdgeDecisionLayer(TieredDecider(tiers, thresholds=DEFAULT_THRESHOLDS), **kw)


def finished(attempt=1):
    return NodeResult("draft_patch", "diff --git ...", "tests pass and diff touches only auth/", attempt)


# --- tiered routing ------------------------------------------------------------


def test_confident_jev_answers_every_question_in_one_call():
    jev = FakeBackend("jev", {"agent": AGENT_CODE, "effort": EFFORT_2, "pass": YES_SURE}, cost=0.0004)
    opus = FakeBackend("opus")
    plan = make_layer(jev, opus).decide(HopInput(graph={"goal": "fix login"}, finished=finished()))

    assert plan.action is Action.PROCEED
    assert (plan.next_agent, plan.effort_level, plan.merge) == ("code", 2, True)
    assert plan.worker.model == "claude-sonnet-5-5"
    assert jev.calls == [("agent", "effort", "pass")]
    assert opus.calls == []
    assert plan.cost_usd == pytest.approx(0.0004)


def test_only_the_unsure_question_goes_up_a_tier():
    jev = FakeBackend("jev", {"agent": {"code": 0.45, "research": 0.4, "write": 0.1, "ask": 0.05}, "effort": EFFORT_2})
    opus = FakeBackend("opus", {"agent": {"research": 0.88, "code": 0.12}}, cost=0.17)
    plan = make_layer(jev, opus).decide(HopInput(graph={}))

    assert opus.calls == [("agent",)]
    assert plan.next_agent == "research"
    assert plan.decision["agent"].tier == "opus"
    assert plan.decision["effort"].tier == "jev"
    assert plan.decision.escalated == ("agent",)


def test_backend_error_escalates_instead_of_crashing():
    jev = FakeBackend("jev", error=TimeoutError("jev timed out"))
    opus = FakeBackend("opus", {"agent": AGENT_CODE, "effort": EFFORT_2})
    plan = make_layer(jev, opus).decide(HopInput(graph={}))

    assert plan.action is Action.PROCEED
    assert plan.decision.trace[0].error.startswith("TimeoutError")
    assert opus.calls == [("agent", "effort")]


def test_out_of_vocabulary_label_is_treated_as_unsure():
    jev = FakeBackend("jev", {"agent": {"deploy": 1.0}, "effort": EFFORT_2})
    opus = FakeBackend("opus", {"agent": AGENT_CODE})
    plan = make_layer(jev, opus).decide(HopInput(graph={}))

    assert plan.next_agent == "code"
    assert any("invalid answer" in n for n in plan.decision["agent"].notes)


# --- the four questions -----------------------------------------------------------


def test_agent_ask_halts_for_the_human():
    jev = FakeBackend("jev", {"agent": {"ask": 0.9, "code": 0.1}, "effort": EFFORT_2})
    plan = make_layer(jev).decide(HopInput(graph={}))
    assert plan.action is Action.HALT_ASK and plan.next_agent == "ask"


def test_unresolved_agent_and_effort_fall_back_conservatively():
    split = {"code": 0.4, "research": 0.35, "write": 0.25}
    effort = {"3": 0.5, "4": 0.5}
    jev = FakeBackend("jev", {"agent": split, "effort": effort})
    opus = FakeBackend("opus", {"agent": split, "effort": effort})
    plan = make_layer(jev, opus).decide(HopInput(graph={}))

    assert plan.action is Action.HALT_ASK and plan.next_agent == "ask"
    assert plan.effort_level == 4  # ceil(3.5): unsure budgets round up
    assert plan.worker.model == "claude-opus-5-5"
    assert not plan.decision["agent"].resolved


def test_clear_unsafe_waits_for_human_without_paying_for_escalation():
    jev = FakeBackend("jev", {"agent": AGENT_CODE, "effort": EFFORT_2, "safe.delete": {"yes": 0.05, "no": 0.95}})
    opus = FakeBackend("opus")
    plan = make_layer(jev, opus).decide(HopInput(graph={}, pending_tool=ToolCall("delete_records")))

    assert plan.action is Action.AWAIT_APPROVAL
    assert plan.blocked_risks == ("delete",)
    assert opus.calls == []


def test_ambiguous_safety_escalates_and_still_holds_if_top_tier_is_unsure():
    jev = FakeBackend("jev", {"agent": AGENT_CODE, "effort": EFFORT_2, "safe.payment": {"yes": 0.6, "no": 0.4}})
    opus = FakeBackend("opus", {"safe.payment": {"yes": 0.9, "no": 0.1}})  # below safe yes_at=0.95
    plan = make_layer(jev, opus).decide(HopInput(graph={}, pending_tool=ToolCall("charge_card")))

    assert opus.calls == [("safe.payment",)]
    assert plan.action is Action.AWAIT_APPROVAL
    assert not plan.decision["safe.payment"].resolved


def test_near_certain_safety_proceeds():
    jev = FakeBackend("jev", {"agent": AGENT_CODE, "effort": EFFORT_2, "safe.outbound_email": YES_SURE})
    plan = make_layer(jev).decide(HopInput(graph={}, pending_tool=ToolCall("send_email")))
    assert plan.action is Action.PROCEED and plan.blocked_risks == ()


def test_each_risk_is_its_own_noul_and_any_block_holds():
    jev = FakeBackend("jev", {
        "agent": AGENT_CODE, "effort": EFFORT_2,
        "safe.delete": YES_SURE, "safe.outbound_email": {"yes": 0.1, "no": 0.9},
    })
    tool = ToolCall("cleanup_and_notify", risks=("delete", "outbound_email"))
    plan = make_layer(jev).decide(HopInput(graph={}, pending_tool=tool))
    assert plan.blocked_risks == ("outbound_email",)


def test_failed_check_reruns_then_reaches_the_human():
    fail = {"yes": 0.1, "no": 0.9}
    jev = FakeBackend("jev", {"agent": AGENT_CODE, "effort": EFFORT_2, "pass": fail})
    layer = make_layer(jev, max_reruns=2)

    first = layer.decide(HopInput(graph={}, finished=finished(attempt=1)))
    assert first.action is Action.RERUN and first.next_agent == "draft_patch" and not first.merge

    last = layer.decide(HopInput(graph={}, finished=finished(attempt=3)))
    assert last.action is Action.HALT_ASK and last.next_agent == "ask"


def test_failed_check_wins_over_a_risky_tool():
    jev = FakeBackend("jev", {
        "agent": AGENT_CODE, "effort": EFFORT_2,
        "pass": {"yes": 0.1, "no": 0.9}, "safe.delete": {"yes": 0.1, "no": 0.9},
    })
    plan = make_layer(jev).decide(HopInput(graph={}, finished=finished(), pending_tool=ToolCall("drop_table")))
    assert plan.action is Action.RERUN


# --- determinism ----------------------------------------------------------------


def test_same_probabilities_always_give_the_same_plan():
    tie = {"research": 0.5, "code": 0.5}
    jev = FakeBackend("jev", {"agent": tie, "effort": {"2": 0.5, "3": 0.5}})
    decider = TieredDecider([jev], thresholds={
        **DEFAULT_THRESHOLDS, "agent": Threshold(min_confidence=0.5), "effort": Threshold(min_confidence=0.5),
    })
    layer = EdgeDecisionLayer(decider)
    plans = {(p.next_agent, p.effort_level) for p in (layer.decide(HopInput(graph={})) for _ in range(50))}
    assert plans == {("research", 2)}  # ties break by declared order


def test_classify_risks():
    assert classify_risks(ToolCall("delete_user")) == ("delete",)
    assert classify_risks(ToolCall("refund_order")) == ("payment",)
    assert classify_risks(ToolCall("read_file")) == ()
    assert classify_risks(ToolCall("read_file", risks=("payment",))) == ("payment",)


def test_no_tool_and_no_finished_node_asks_only_two_questions():
    layer = make_layer(FakeBackend("jev"))
    assert list(layer.questions_for(HopInput(graph={}))) == ["agent", "effort"]


def test_normalize_rejects_bad_distributions():
    with pytest.raises(InvalidDistribution):
        normalize(Noul("?"), {"maybe": 1.0})
    with pytest.raises(InvalidDistribution):
        normalize(Choice(("a", "b")), {"a": 0.0, "b": 0.0})
    assert normalize(Choice(("a", "b")), {"a": 3, "b": 1}) == {"a": 0.75, "b": 0.25}


# --- backends -----------------------------------------------------------------------


def test_jev_backend_matches_the_working_note_call_shape():
    seen = {}

    class Client:
        def decide(self, model, state, questions):
            seen.update(model=model, state=state, questions=questions)
            return {"agent": {"probs": AGENT_CODE}, "usage": {"input_tokens": 10_000}}

    qs = {"agent": Choice(("research", "code", "write", "ask"))}
    result = JevBackend(Client()).decide({"g": 1}, qs)
    assert seen["model"] == "jev-latest" and seen["questions"] == qs
    assert result.distributions["agent"] == AGENT_CODE
    assert result.cost_usd == pytest.approx(10_000 * 0.042 / 1e6)


def _claude_client(stop_reason="end_turn", payload=None):
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        content = [SimpleNamespace(type="thinking", thinking="")]
        if payload is not None:
            content.append(SimpleNamespace(type="text", text=json.dumps(payload)))
        return SimpleNamespace(
            stop_reason=stop_reason, content=content,
            usage=SimpleNamespace(input_tokens=2_000, output_tokens=100),
        )

    client = SimpleNamespace(beta=SimpleNamespace(messages=SimpleNamespace(create=create)))
    return client, calls


def test_claude_backend_request_and_parse():
    client, calls = _claude_client(payload={"agent": AGENT_CODE})
    qs = {"agent": Choice(("research", "code", "write", "ask"))}
    result = ClaudeBackend(client).decide({"b": 2, "a": 1}, qs)

    req = calls[0]
    assert req["model"] == "claude-opus-5-5"
    assert req["fallbacks"] == "default" and req["betas"] == ["server-side-fallback-2026-07-01"]
    schema = req["output_config"]["format"]["schema"]
    assert schema["properties"]["agent"]["required"] == ["research", "code", "write", "ask"]
    assert '{"a": 1, "b": 2}' in req["messages"][0]["content"]  # sorted, byte-stable state
    assert "thinking" not in req
    assert result.distributions["agent"] == AGENT_CODE
    assert result.cost_usd == pytest.approx((2_000 * 4 + 100 * 20) / 1e6)


def test_claude_refusal_leaves_question_unresolved_and_policy_holds():
    client, _ = _claude_client(stop_reason="refusal")
    with pytest.raises(BackendError):
        ClaudeBackend(client).decide({}, {"pass": Noul("?")})

    jev = FakeBackend("jev", {"agent": AGENT_CODE, "effort": EFFORT_2, "safe.delete": {"yes": 0.5, "no": 0.5}})
    layer = make_layer(jev, ClaudeBackend(client, name="opus"))
    plan = layer.decide(HopInput(graph={}, pending_tool=ToolCall("delete_branch")))
    assert plan.action is Action.AWAIT_APPROVAL
    assert plan.decision.trace[1].error.startswith("BackendError")

import json

from jev_edges import DEFAULT_THRESHOLDS, DecideResult, DecisionLog, EdgeDecisionLayer, HopInput, TieredDecider, ToolCall
from jev_edges.calibrate import Sample, load_samples, main, suggest
from jev_edges.safety import SAFE_YES_FLOOR


class Fixed:
    def __init__(self, name, answers):
        self.name, self.answers = name, answers

    def decide(self, state, questions):
        return DecideResult({k: self.answers[k] for k in questions if k in self.answers})


def test_log_keeps_every_tier_but_no_content(tmp_path):
    jev = Fixed("jev", {"agent": {"code": 0.5, "research": 0.5}, "effort": {"2": 1.0},
                        "safe.delete": {"yes": 0.99, "no": 0.01}})
    opus = Fixed("opus", {"agent": {"research": 0.9, "code": 0.1}})
    layer = EdgeDecisionLayer(TieredDecider([jev, opus], thresholds=DEFAULT_THRESHOLDS))
    plan = layer.decide(HopInput(graph={"secret": "customer-db-password"},
                                 pending_tool=ToolCall("delete_rows", {"table": "users"})))

    log = DecisionLog(tmp_path / "d.jsonl", clock=lambda: 0.0)
    log.record("h1", plan)
    log.record_outcome("h1", "agent", "research")
    raw = (tmp_path / "d.jsonl").read_text()

    assert "customer-db-password" not in raw and "users" not in raw
    rec = json.loads(raw.splitlines()[0])
    assert [h["tier"] for h in rec["answers"]["agent"]["history"]] == ["jev", "opus"]
    assert load_samples(log, "jev") == [Sample("agent", {"code": 0.5, "research": 0.5, "write": 0.0, "ask": 0.0}, "research")]


def _choice(conf, right):
    other = 1 - conf
    probs = {"code": conf, "research": other}
    return Sample("agent", probs, "code" if right else "research")


def test_choice_threshold_is_the_lowest_that_meets_the_target():
    samples = [_choice(0.95, True)] * 40 + [_choice(0.75, True)] * 30 + [_choice(0.6, False)] * 30
    out = suggest(samples, {"agent": 0.9}, min_samples=30)["agent"]["suggestion"]
    assert out["min_confidence"] == 0.75
    assert out["escalation_rate"] == 0.3


def test_noul_band_and_safety_clamp():
    samples = ([Sample("safe", {"yes": 0.88, "no": 0.12}, "yes")] * 40
               + [Sample("safe", {"yes": 0.1, "no": 0.9}, "no")] * 40)
    out = suggest(samples, {"safe": 0.99}, min_samples=30)["safe"]["suggestion"]
    assert out["yes_at"] == SAFE_YES_FLOOR and out["clamped_to_floor"]
    assert out["no_at"] == 0.1


def test_too_little_data_keeps_current_threshold():
    out = suggest([_choice(0.9, True)] * 5, min_samples=30)["agent"]
    assert out["suggestion"] is None and "keep the current" in out["note"]


def test_cli_prints_json(tmp_path, capsys):
    log = DecisionLog(tmp_path / "d.jsonl")
    main([str(log.path)])
    assert json.loads(capsys.readouterr().out) == {}

"""Guard tests: these fail CI if the safety policy is weakened."""
import pytest

from jev_edges import (
    DEFAULT_THRESHOLDS, EdgeDecisionLayer, Threshold, TieredDecider, UnsafeConfiguration,
)
from jev_edges.safety import RISK_RULES, SAFE_THRESHOLD, SAFE_YES_FLOOR


class Null:
    name = "null"

    def decide(self, state, questions):
        raise AssertionError("not called")


def test_policy_is_not_weakened():
    assert SAFE_YES_FLOOR >= 0.90
    assert SAFE_THRESHOLD.yes_at >= 0.95
    assert DEFAULT_THRESHOLDS["safe"] is SAFE_THRESHOLD
    assert {"delete", "payment", "outbound_email"} <= set(RISK_RULES)


def test_layer_refuses_a_weak_safety_threshold():
    weak = {**DEFAULT_THRESHOLDS, "safe": Threshold(yes_at=0.7, no_at=0.2)}
    with pytest.raises(UnsafeConfiguration):
        EdgeDecisionLayer(TieredDecider([Null()], thresholds=weak))


def test_layer_refuses_a_weak_per_risk_override():
    weak = {**DEFAULT_THRESHOLDS, "safe.payment": Threshold(yes_at=0.85, no_at=0.2)}
    with pytest.raises(UnsafeConfiguration):
        EdgeDecisionLayer(TieredDecider([Null()], thresholds=weak))


def test_layer_refuses_an_implicit_default_for_safety():
    with pytest.raises(UnsafeConfiguration):
        EdgeDecisionLayer(TieredDecider([Null()]))  # Threshold() default yes_at=0.8

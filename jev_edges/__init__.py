"""Deterministic edge decision layer for agent graphs (after "Jev Engineering for Agent Graphs").

    decider = TieredDecider([JevBackend(jev_client), ClaudeBackend(anthropic_client)],
                            thresholds=DEFAULT_THRESHOLDS)
    layer = EdgeDecisionLayer(decider)
    plan = layer.decide(HopInput(graph=state, finished=..., pending_tool=...))
"""
from .backends import BackendError, ClaudeBackend, DecideResult, DecisionBackend, JevBackend
from .layer import (
    DEFAULT_AGENTS,
    DEFAULT_EFFORT_RUBRIC,
    DEFAULT_RISK_RULES,
    DEFAULT_THRESHOLDS,
    DEFAULT_WORKERS,
    Action,
    EdgeDecisionLayer,
    HopInput,
    HopPlan,
    NodeResult,
    ToolCall,
    WorkerConfig,
    classify_risks,
)
from .log import DecisionLog
from .questions import Answer, Choice, Noul, Question, Score
from .router import Decision, Threshold, TieredDecider, TierCall
from .safety import SAFE_THRESHOLD, SAFE_YES_FLOOR, UnsafeConfiguration

__all__ = [
    "Action", "Answer", "BackendError", "Choice", "ClaudeBackend", "DEFAULT_AGENTS",
    "DEFAULT_EFFORT_RUBRIC", "DEFAULT_RISK_RULES", "DEFAULT_THRESHOLDS", "DEFAULT_WORKERS",
    "DecideResult", "Decision", "DecisionBackend", "DecisionLog", "EdgeDecisionLayer", "HopInput",
    "HopPlan", "JevBackend", "NodeResult", "Noul", "Question", "SAFE_THRESHOLD", "SAFE_YES_FLOOR",
    "Score", "Threshold", "TierCall", "TieredDecider", "ToolCall", "UnsafeConfiguration",
    "WorkerConfig", "classify_risks",
]

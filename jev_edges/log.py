"""Append-only JSONL log of edge decisions and their later ground truth."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable, Iterator, Union

from .layer import HopPlan


class DecisionLog:
    """Records what each tier said on every hop, then what turned out to be right.

    Only probabilities, tiers and actions are written - never graph state, tool
    arguments or node output - so the log can be kept for calibration without
    carrying the content the agents worked on.

    Truth values per question family:
      agent  -> the label that should have run      effort -> the level that sufficed ("1".."5")
      safe.* -> "yes" if the action was safe/undoable, "no" otherwise
      pass   -> "yes" if the output really met its check
    """

    def __init__(self, path: Union[str, Path], clock: Callable[[], float] = time.time) -> None:
        self.path = Path(path)
        self.clock = clock

    def _append(self, record: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")

    def record(self, hop_id: str, plan: HopPlan) -> None:
        self._append({
            "kind": "decision",
            "hop_id": hop_id,
            "ts": self.clock(),
            "action": plan.action.value,
            "next_agent": plan.next_agent,
            "effort_level": plan.effort_level,
            "blocked_risks": list(plan.blocked_risks),
            "cost_usd": plan.cost_usd,
            "answers": {
                key: {
                    "tier": a.tier,
                    "resolved": a.resolved,
                    "top": a.top if a.probs else None,
                    "history": [{"tier": t, "probs": dict(p)} for t, p in a.history],
                }
                for key, a in plan.decision.answers.items()
            },
        })

    def record_outcome(self, hop_id: str, key: str, truth: str) -> None:
        self._append({"kind": "outcome", "hop_id": hop_id, "key": key, "truth": str(truth), "ts": self.clock()})

    def __iter__(self) -> Iterator[dict]:
        if not self.path.exists():
            return
        with self.path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)

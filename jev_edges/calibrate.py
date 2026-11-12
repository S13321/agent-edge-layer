"""Suggest per-question thresholds from a DecisionLog.

    python -m jev_edges.calibrate decisions.jsonl --tier jev --min-samples 30

Suggestions are printed, never applied: threshold changes go through review like
any other code change, and safety suggestions are clamped to the safety floor.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import Iterable, Mapping, Optional

from .safety import SAFE_YES_FLOOR

DEFAULT_TARGETS = {"agent": 0.90, "effort": 0.90, "pass": 0.90, "safe": 0.99}


@dataclass(frozen=True)
class Sample:
    family: str
    probs: Mapping[str, float]
    truth: str

    @property
    def is_noul(self) -> bool:
        return set(self.probs) == {"yes", "no"}

    @property
    def top(self) -> str:
        return max(sorted(self.probs), key=lambda o: self.probs[o])


def load_samples(records: Iterable[dict], tier: str = "jev") -> list[Sample]:
    """Pair each outcome with what `tier` said for that question, escalated or not."""
    records = list(records)
    decisions = {r["hop_id"]: r for r in records if r.get("kind") == "decision"}
    samples = []
    for r in records:
        if r.get("kind") != "outcome":
            continue
        answer = decisions.get(r["hop_id"], {}).get("answers", {}).get(r["key"])
        if not answer:
            continue
        said = next((h["probs"] for h in answer["history"] if h["tier"] == tier), None)
        if said:
            samples.append(Sample(r["key"].split(".", 1)[0], said, r["truth"]))
    return samples


def _min_confidence(samples: list[Sample], target: float, min_n: int) -> Optional[dict]:
    for t in sorted({max(s.probs.values()) for s in samples}):
        covered = [s for s in samples if max(s.probs.values()) >= t]
        if len(covered) < min_n:
            break
        acc = sum(s.top == s.truth for s in covered) / len(covered)
        if acc >= target:
            return {"min_confidence": round(t, 4), "accuracy": round(acc, 4),
                    "escalation_rate": round(1 - len(covered) / len(samples), 4)}
    return None


def _noul_band(samples: list[Sample], target: float, min_n: int, yes_floor: float = 0.0) -> dict:
    ps = sorted({s.probs["yes"] for s in samples})
    out: dict = {}
    for t in ps:  # smallest yes_at whose "settled yes" set is precise enough
        covered = [s for s in samples if s.probs["yes"] >= t]
        if len(covered) < min_n:
            break
        prec = sum(s.truth == "yes" for s in covered) / len(covered)
        if prec >= target:
            out.update(yes_at=round(t, 4), yes_precision=round(prec, 4))
            break
    for t in reversed(ps):  # largest no_at whose "settled no" set is precise enough
        covered = [s for s in samples if s.probs["yes"] <= t]
        if len(covered) < min_n:
            break
        prec = sum(s.truth == "no" for s in covered) / len(covered)
        if prec >= target and t < out.get("yes_at", 1.0):
            out.update(no_at=round(t, 4), no_precision=round(prec, 4))
            break
    if "yes_at" in out and out["yes_at"] < yes_floor:
        out["yes_at"] = yes_floor
        out["clamped_to_floor"] = True
    if "yes_at" in out:
        lo = out.get("no_at", -1.0)
        settled = [s for s in samples if s.probs["yes"] >= out["yes_at"] or s.probs["yes"] <= lo]
        out["escalation_rate"] = round(1 - len(settled) / len(samples), 4)
    return out


def suggest(samples: list[Sample], targets: Mapping[str, float] = DEFAULT_TARGETS, min_samples: int = 30) -> dict:
    report: dict = {}
    for family in sorted({s.family for s in samples}):
        group = [s for s in samples if s.family == family]
        target = targets.get(family, 0.90)
        entry: dict = {"n": len(group), "target": target}
        if all(s.is_noul for s in group):
            floor = SAFE_YES_FLOOR if family == "safe" else 0.0
            entry["suggestion"] = _noul_band(group, target, min_samples, floor) or None
        else:
            entry["suggestion"] = _min_confidence(group, target, min_samples)
        if entry["suggestion"] is None:
            entry["note"] = "not enough samples or target unreachable; keep the current threshold"
        report[family] = entry
    return report


def main(argv: Optional[list[str]] = None) -> None:
    from .log import DecisionLog

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("log")
    ap.add_argument("--tier", default="jev")
    ap.add_argument("--min-samples", type=int, default=30)
    args = ap.parse_args(argv)
    samples = load_samples(DecisionLog(args.log), args.tier)
    print(json.dumps(suggest(samples, min_samples=args.min_samples), indent=2))


if __name__ == "__main__":
    main()

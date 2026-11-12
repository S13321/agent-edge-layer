"""Safety policy for irreversible actions.

Every change to this file needs code-owner review (see .github/CODEOWNERS), and
tests/test_safety.py fails CI if the policy is weakened.
"""
from __future__ import annotations

from typing import Mapping

from .router import Threshold

# Deterministic tool-name -> risk tags. A ToolCall's explicit `risks` wins over these.
RISK_RULES: Mapping[str, tuple[str, ...]] = {
    "delete": ("delete", "remove", "drop", "truncate", "destroy", "rm_"),
    "payment": ("pay", "charge", "refund", "transfer", "invoice", "purchase"),
    "outbound_email": ("send_email", "send_mail", "email_send", "smtp", "outbound"),
}

# Proceed only on near-certainty; a clear "no" waits for a human without paying to escalate.
SAFE_THRESHOLD = Threshold(yes_at=0.95, no_at=0.20)

# No configuration may clear an irreversible action below this p(safe), calibrated or not.
SAFE_YES_FLOOR = 0.90


class UnsafeConfiguration(ValueError):
    pass


def check_safe_thresholds(thresholds: Mapping[str, Threshold], fallback: Threshold) -> None:
    """Reject any safety threshold under the floor, including the implicit default."""
    family = thresholds.get("safe", fallback)
    checked = {"safe": family, **{k: t for k, t in thresholds.items() if k.startswith("safe.")}}
    weak = {k: t.yes_at for k, t in checked.items() if t.yes_at < SAFE_YES_FLOOR}
    if weak:
        raise UnsafeConfiguration(
            f"safety thresholds below floor {SAFE_YES_FLOOR}: {weak}. "
            "Configure 'safe' explicitly (e.g. safety.SAFE_THRESHOLD)."
        )

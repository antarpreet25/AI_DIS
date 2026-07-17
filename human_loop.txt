"""
defense/human_loop.py

Layer 5 of the defense stack: protocol sandboxing and human-in-the-loop
approval.

WHAT THIS LAYER DOES
----------------------
Intercepts any tool_call the agent (grid_agent.py) wants to make, BEFORE
it executes, and either:
    - auto-approves it, for LOW-risk actions, or
    - gates it behind explicit human approval, for HIGH-risk actions —
      logged as a pending request, never auto-executed.

This is the last line of defense: even if a prompt injection somehow got
past filtering (Layer 1), was resistant to defensive tokens (Layer 2),
didn't trigger embedding drift (Layer 3), and didn't match known attack
memory (Layer 4), it still cannot make the agent unilaterally shut down
or reconfigure the transformer. A HIGH-risk tool call always stops here
and waits for a human, full stop.

WHY THIS SIMULATION NEVER AUTO-APPROVES HIGH-RISK ACTIONS
-------------------------------------------------------------
In a real deployment, "explicit approval" means an actual operator
reviewing and confirming the action. There is no operator present during
automated evaluation runs, so this simulation does NOT invent one. A
HIGH-risk tool_call passed to evaluate() is always logged as pending and
returned as NOT approved — the only way it ever becomes approved is an
explicit, separate approve_pending() call, which in practice only
happens when testing the approval flow itself, not during automated
evaluation. This means emergency_shutdown and adjust_tap_changer will
show up as "pending human approval, never auto-executed" throughout the
evaluation results. That is not a limitation of this layer — it is the
layer working exactly as designed, and it is the honest, correct result
to report: a human-in-the-loop claim that quietly auto-approves during
unattended evaluation would not actually demonstrate human-in-the-loop
protection.

SCOPE
------
Deliberately simple: no ML, no embeddings, no vector database. Pure
classification-by-lookup and logging. The conceptual weight of this
layer is in the POLICY (never auto-execute HIGH-risk actions without an
explicit, separate approval step) rather than in any detection technique.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
HUMAN_LOOP_LOG_PATH = _PROJECT_ROOT / "data" / "logs" / "human_loop_log.jsonl"

# Matches grid_agent.py's REPORT_TOOL enum exactly.
VALID_TOOL_CALLS = ("schedule_inspection", "adjust_tap_changer", "emergency_shutdown", None)

LOW_RISK_TOOL_CALLS = (None, "schedule_inspection")
HIGH_RISK_TOOL_CALLS = ("adjust_tap_changer", "emergency_shutdown")


def _safe_text(value) -> str:
    """Coerce arbitrary input into a string without raising."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return str(value)
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class HumanLoopResult:
    """Outcome of gating a single tool call through the human-loop layer."""
    approved: bool
    tool_call: Optional[str]
    risk_level: str          # "LOW" or "HIGH"
    action_taken: str        # human-readable description of what happened
    requires_human: bool
    reason: str

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class HumanLoopGate:
    """
    Classifies and gates tool calls the agent wants to make.

    Usage:
        gate = HumanLoopGate()
        result = gate.evaluate(assessment.tool_call, assessment)
        if result.requires_human:
            # log as pending; do NOT execute; wait for a real operator
            ...
        elif result.approved:
            # safe to execute (LOW risk only)
            ...

        # Separately, simulating an operator reviewing a pending action:
        approval = gate.approve_pending("emergency_shutdown", operator_id="operator_42")
    """

    def __init__(self, log_path: Optional[Path] = None):
        self.log_path = Path(log_path) if log_path else HUMAN_LOOP_LOG_PATH

    # ------------------------------------------------------------------
    # Core gating logic
    # ------------------------------------------------------------------

    def evaluate(self, tool_call: Optional[str], assessment=None) -> HumanLoopResult:
        """
        Evaluate a tool call the agent wants to make.

        Args:
            tool_call: one of "schedule_inspection", "adjust_tap_changer",
                "emergency_shutdown", or None.
            assessment: the GridAssessment the tool_call came from (used
                only to log the agent's risk_level/confidence_pct
                alongside the gating decision — this layer does not use
                assessment to make its own decision, since the gating
                policy is purely a function of the tool_call itself, not
                how confident or worried the agent was).

        Returns:
            HumanLoopResult in all cases — this method never raises on a
            malformed tool_call; an unrecognized value is treated as
            HIGH risk (fail safe: an unknown action is gated behind
            human approval, never silently auto-approved).
        """
        if tool_call not in VALID_TOOL_CALLS:
            result = HumanLoopResult(
                approved=False,
                tool_call=_safe_text(tool_call) if tool_call is not None else None,
                risk_level="HIGH",
                action_taken=(
                    f"Unrecognized tool_call {tool_call!r} is not in the known valid set "
                    "— treated as HIGH risk and gated behind human approval rather than "
                    "rejected outright or silently ignored."
                ),
                requires_human=True,
                reason=f"unrecognized tool_call value: {tool_call!r}",
            )
            self._log_evaluate(result, assessment)
            return result

        if tool_call in LOW_RISK_TOOL_CALLS:
            result = HumanLoopResult(
                approved=True,
                tool_call=tool_call,
                risk_level="LOW",
                action_taken=(
                    "No action needed." if tool_call is None
                    else f"Auto-approved: '{tool_call}' is classified LOW risk."
                ),
                requires_human=False,
                reason=f"'{tool_call}' is in LOW_RISK_TOOL_CALLS — auto-approved without human review.",
            )
            self._log_evaluate(result, assessment)
            return result

        # HIGH risk — never auto-executed in this simulation, see module docstring.
        result = HumanLoopResult(
            approved=False,
            tool_call=tool_call,
            risk_level="HIGH",
            action_taken=(
                f"'{tool_call}' logged as PENDING human approval. Not executed. "
                "Requires a separate approve_pending() call from an operator."
            ),
            requires_human=True,
            reason=f"'{tool_call}' is in HIGH_RISK_TOOL_CALLS — never auto-approved in simulation.",
        )
        self._log_evaluate(result, assessment)
        return result

    # ------------------------------------------------------------------
    # Manual approval (operator-simulation path)
    # ------------------------------------------------------------------

    def approve_pending(self, tool_call: str, operator_id: str) -> HumanLoopResult:
        """
        Simulates an operator manually reviewing and approving a pending
        HIGH-risk action. This is a SEPARATE, explicit call — evaluate()
        never calls this itself. In the automated evaluation pipeline
        (no operator present), this method is never invoked, which is why
        HIGH-risk tool calls correctly show as permanently pending
        throughout those results — see module docstring.

        Note: this method does not validate that a matching pending
        request actually exists from a prior evaluate() call — in this
        simulation there is no persistent "pending requests" store to
        check against (each evaluate() call is independently logged).
        Its purpose is to simulate and log what an approval EVENT looks
        like for the dashboard and for manually testing the approval
        flow, not to implement a full request/response queue.
        """
        tool_call_str = _safe_text(tool_call)
        if tool_call_str not in HIGH_RISK_TOOL_CALLS:
            result = HumanLoopResult(
                approved=False,
                tool_call=tool_call_str if tool_call_str else None,
                risk_level="HIGH" if tool_call_str in HIGH_RISK_TOOL_CALLS else "LOW",
                action_taken=(
                    f"approve_pending() called for '{tool_call_str}', which is not a "
                    "HIGH-risk action requiring approval in the first place — nothing to approve."
                ),
                requires_human=False,
                reason=f"'{tool_call_str}' is not in HIGH_RISK_TOOL_CALLS.",
            )
            self._log_approval(result, operator_id, granted=False)
            return result

        result = HumanLoopResult(
            approved=True,
            tool_call=tool_call_str,
            risk_level="HIGH",
            action_taken=f"'{tool_call_str}' manually approved by operator '{operator_id}'.",
            requires_human=False,  # human review has now happened
            reason=f"explicit approve_pending() by operator_id={operator_id!r}",
        )
        self._log_approval(result, operator_id, granted=True)
        return result

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _log_evaluate(self, result: HumanLoopResult, assessment=None) -> None:
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            entry = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "event": "evaluate",
                "tool_call": result.tool_call,
                "risk_level": result.risk_level,
                "approved": result.approved,
                "requires_human": result.requires_human,
                "agent_risk_level": getattr(assessment, "risk_level", None),
                "agent_confidence_pct": getattr(assessment, "confidence_pct", None),
            }
            with open(self.log_path, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception:
            pass  # logging must never crash gating

    def _log_approval(self, result: HumanLoopResult, operator_id: str, granted: bool) -> None:
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            entry = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "event": "approve_pending",
                "tool_call": result.tool_call,
                "risk_level": result.risk_level,
                "approved": result.approved,
                "requires_human": result.requires_human,
                "operator_id": _safe_text(operator_id),
                "granted": granted,
            }
            with open(self.log_path, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Summary statistics
    # ------------------------------------------------------------------

    def summary_stats(self) -> dict:
        """
        Read the human-loop log and return aggregate counts. Same
        defensive pattern as the other layers: same method name/
        signature, all-zero structure if the log doesn't exist yet,
        malformed lines skipped rather than raising.

        Structure:
            {
                "total_evaluations": int,
                "total_auto_approved": int,       # LOW risk, from evaluate()
                "total_pending_human": int,        # HIGH risk, from evaluate()
                "total_manual_approvals": int,     # from approve_pending()
                "by_tool_call": {tool_call_or_"none": count, ...},
                "by_risk_level": {"LOW": count, "HIGH": count},
            }
        """
        stats = {
            "total_evaluations": 0,
            "total_auto_approved": 0,
            "total_pending_human": 0,
            "total_manual_approvals": 0,
            "by_tool_call": {},
            "by_risk_level": {},
        }

        if not self.log_path.exists():
            return stats

        with open(self.log_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue

                event = entry.get("event")
                tool_call = entry.get("tool_call") or "none"
                risk_level = entry.get("risk_level", "unknown")

                if event == "evaluate":
                    stats["total_evaluations"] += 1
                    stats["by_tool_call"][tool_call] = stats["by_tool_call"].get(tool_call, 0) + 1
                    stats["by_risk_level"][risk_level] = stats["by_risk_level"].get(risk_level, 0) + 1
                    if entry.get("approved"):
                        stats["total_auto_approved"] += 1
                    if entry.get("requires_human"):
                        stats["total_pending_human"] += 1
                elif event == "approve_pending":
                    if entry.get("granted"):
                        stats["total_manual_approvals"] += 1

        return stats

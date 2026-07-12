"""Verifier — Phase 9 operational guardrail.

Validates the LLM Planner's structured action decision before it is converted
into a Lesson and applied to the policy. Catches:
  - Invalid action names
  - Invalid parameter values
  - Logically impossible actions (e.g. SWITCH_MODEL at the bottom of hierarchy)
  - Overly aggressive threshold calibration

If verification fails, the verifier falls back to the rule-based SelfReflector
and logs the rejection reason. This ensures the agent never takes a destructive
or incoherent action due to LLM hallucination.
"""

from __future__ import annotations

from dataclasses import dataclass

from agent import agent_config as config
from agent.episodic_memory import (
    ALL_LESSON_ACTIONS,
    CALIBRATE_THRESHOLD,
    REINFORCE,
    SWITCH_MODEL,
    SWITCH_SUBSET,
    SWITCH_TYPE,
)
from agent.tool_library import validate_action


@dataclass
class VerificationResult:
    ok:             bool
    action:         str
    params:         dict
    rejection_reason: str = ""

    @property
    def failed(self) -> bool:
        return not self.ok


def verify(
    plan:   dict,
    policy: dict,
    report_auroc:    float,
    report_drift:    bool,
    report_model:    str,
) -> VerificationResult:
    """Verify the LLM Planner's decision against safety and logic constraints.

    Args:
        plan:         Output of LLMPlanner.plan() with keys action, params.
        policy:       Current EpisodicMemory policy dict.
        report_auroc: AUROC from the current EpisodeReport.
        report_drift: drift_detected from the current EpisodeReport.
        report_model: model_used from the current EpisodeReport.

    Returns:
        VerificationResult — ok=True if safe to proceed, rejection_reason if not.
    """
    action = plan.get("action", "")
    params = plan.get("params") or {}

    # ── 1. Schema validation ─────────────────────────────────────────────────
    ok, err = validate_action(action, params)
    if not ok:
        return VerificationResult(ok=False, action=action, params=params, rejection_reason=err)

    # ── 2. Logic constraints ─────────────────────────────────────────────────

    if action == REINFORCE and report_auroc < config.AUROC_FLOOR:
        return VerificationResult(
            ok=False, action=action, params=params,
            rejection_reason=(
                f"REINFORCE rejected: AUROC={report_auroc:.4f} is below floor "
                f"{config.AUROC_FLOOR}. Cannot reinforce a failing model."
            ),
        )

    if action == SWITCH_SUBSET and not report_drift:
        # Allow but warn — SWITCH_SUBSET without drift is unusual but not fatal
        # (could be pro-active retraining). Let it through.
        pass

    if action == SWITCH_MODEL:
        h = policy.get("model_hierarchy", [])
        from reasoning.selector import TIER_TO_NAME
        name_to_tier = {v: k for k, v in TIER_TO_NAME.items()}
        tier = name_to_tier.get(report_model)
        if tier and tier in h:
            idx = h.index(tier)
            if idx >= len(h) - 1:
                return VerificationResult(
                    ok=False, action=action, params=params,
                    rejection_reason=(
                        f"SWITCH_MODEL rejected: already at the bottom of hierarchy "
                        f"(tier={tier}, hierarchy={h}). Cannot demote further."
                    ),
                )

    if action == CALIBRATE_THRESHOLD:
        current_threshold = policy.get("decision_threshold", 0.5)
        direction = params.get("direction", "lower")
        magnitude = float(params.get("magnitude", 0.05))
        new_threshold = (
            current_threshold - magnitude if direction == "lower"
            else current_threshold + magnitude
        )
        if not (0.10 <= new_threshold <= 0.90):
            return VerificationResult(
                ok=False, action=action, params=params,
                rejection_reason=(
                    f"CALIBRATE_THRESHOLD rejected: new threshold {new_threshold:.2f} "
                    f"would be out of safe range [0.10, 0.90]."
                ),
            )

    if action == SWITCH_TYPE and report_auroc < config.AUROC_FLOOR:
        # Stage 1 is still failing — fix Stage 1 first before typing
        return VerificationResult(
            ok=False, action=action, params=params,
            rejection_reason=(
                f"SWITCH_TYPE rejected: Stage 1 AUROC={report_auroc:.4f} is below "
                f"floor. Fix Stage 1 first (SWITCH_SUBSET or SWITCH_MODEL)."
            ),
        )

    return VerificationResult(ok=True, action=action, params=params)

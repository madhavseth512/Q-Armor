"""SelfReflector: heuristic rules engine for the Reflexion outer loop.

Reads the most recent EpisodeReport(s) from EpisodicMemory and writes a
Lesson — or returns None if the episode was healthy. The Lesson is then
consumed by EpisodicMemory.record() which updates the policy.

Four rules (evaluated in priority order):
  1. BINARY_ONLY   — macro-F1 < MACRO_F1_FLOOR  (multi-class has collapsed)
  2. SWITCH_SUBSET — AUROC < AUROC_FLOOR AND drift detected
  3. SWITCH_MODEL  — AUROC < AUROC_FLOOR AND no drift  (kernel over-specialised)
  4. REINFORCE     — AUROC >= AUROC_FLOOR for REINFORCE_CONSECUTIVE episodes
"""

from __future__ import annotations

from datetime import datetime, timezone

from agent import agent_config as config
from agent.episodic_memory import (
    BINARY_ONLY,
    NO_LESSON,
    REINFORCE,
    SWITCH_MODEL,
    SWITCH_SUBSET,
    EpisodicMemory,
    Lesson,
)
from agent.evaluator import EpisodeReport
from reasoning.selector import TIER_TO_NAME

# Reverse map: model name → hierarchy tier label (e.g. "pegasos_qsvc" → "QSVM")
_NAME_TO_TIER: dict[str, str] = {v: k for k, v in TIER_TO_NAME.items()}


def _actual_tier(model_used: str, hierarchy: list[str]) -> str:
    """Return the tier label for the model that actually ran this episode.

    Falls back to h[0] when the model name is not found in TIER_TO_NAME
    (e.g. a custom or future model not yet in the registry).
    """
    tier = _NAME_TO_TIER.get(model_used)
    if tier and tier in hierarchy:
        return tier
    return hierarchy[0] if hierarchy else "UNKNOWN"


class SelfReflector:
    """Stateless heuristic engine — all state lives in EpisodicMemory."""

    def reflect(
        self,
        report:  EpisodeReport,
        memory:  EpisodicMemory,
        policy:  dict,
    ) -> Lesson | None:
        """Inspect the latest episode report and emit a Lesson or None.

        Args:
            report:  The EpisodeReport just produced by the Evaluator.
            memory:  EpisodicMemory instance (used to read recent history).
            policy:  Current policy dict (used to know the active model tier).

        Returns:
            A Lesson if a rule fires, else None.
        """
        ts = datetime.now(timezone.utc).isoformat()

        # ── Rule 1: BINARY_ONLY — multi-class has collapsed ──────────────────
        # Only triggered when not already in binary-only mode.
        if not policy.get("binary_only", False):
            # We don't have macro_f1 in EpisodeReport (binary episode).
            # The reflector is called from the multi-class evaluator path;
            # the experiment passes macro_f1 via report.binary_f1 when calling
            # reflect() for the multi-class case.  See phase7_reflexion.py.
            # Guard: treat binary_f1 as macro_f1 proxy only when model_used
            # contains 'multiclass' tag, otherwise skip this rule.
            if "multiclass" in report.model_used and report.binary_f1 < config.MACRO_F1_FLOOR:
                return Lesson(
                    lesson_id  = memory.next_lesson_id(),
                    episode_id = report.episode_id,
                    action     = BINARY_ONLY,
                    trigger    = (
                        f"macro-F1={report.binary_f1:.4f} < MACRO_F1_FLOOR="
                        f"{config.MACRO_F1_FLOOR} — multi-class classifier collapsed"
                    ),
                    params     = {
                        "macro_f1":       report.binary_f1,
                        "macro_f1_floor": config.MACRO_F1_FLOOR,
                    },
                    timestamp  = ts,
                )

        # ── Rule 2: SWITCH_SUBSET — low AUROC + drift ────────────────────────
        if report.auroc < config.AUROC_FLOOR and report.drift_detected:
            h = policy["model_hierarchy"]
            used_tier = _actual_tier(report.model_used, h)
            return Lesson(
                lesson_id  = memory.next_lesson_id(),
                episode_id = report.episode_id,
                action     = SWITCH_SUBSET,
                trigger    = (
                    f"AUROC={report.auroc:.4f} < {config.AUROC_FLOOR} WITH drift "
                    f"— feature distribution shifted, re-select kernel subset"
                ),
                params     = {
                    "auroc":        report.auroc,
                    "auroc_floor":  config.AUROC_FLOOR,
                    "model":        used_tier,
                },
                timestamp  = ts,
            )

        # ── Rule 3: SWITCH_MODEL — low AUROC, no drift ───────────────────────
        # Use the tier of the model that *actually ran* (not h[0], which may be
        # an unregistered tier that was skipped by the model selector).
        if report.auroc < config.AUROC_FLOOR and not report.drift_detected:
            h = policy["model_hierarchy"]
            used_tier = _actual_tier(report.model_used, h)
            used_idx  = h.index(used_tier) if used_tier in h else 0
            next_tier = h[used_idx + 1] if used_idx + 1 < len(h) else h[used_idx]
            return Lesson(
                lesson_id  = memory.next_lesson_id(),
                episode_id = report.episode_id,
                action     = SWITCH_MODEL,
                trigger    = (
                    f"AUROC={report.auroc:.4f} < {config.AUROC_FLOOR} WITHOUT drift "
                    f"— kernel over-specialised on training distribution; try next tier"
                ),
                params     = {
                    "auroc":          report.auroc,
                    "auroc_floor":    config.AUROC_FLOOR,
                    "demoted_model":  used_tier,
                    "next_model":     next_tier,
                },
                timestamp  = ts,
            )

        # ── Rule 4: REINFORCE — REINFORCE_CONSECUTIVE healthy episodes ────────
        # The current report is not yet stored, so we need n-1 healthy stored
        # records plus the current healthy report to complete the streak.
        n_consec = config.REINFORCE_CONSECUTIVE
        n_prior = n_consec - 1
        prior = memory.recent_reports(n_prior)
        if (
            len(prior) >= n_prior
            and report.auroc >= config.AUROC_FLOOR
            and all(r.auroc >= config.AUROC_FLOOR for r in prior)
        ):
            h = policy["model_hierarchy"]
            used_tier = _actual_tier(report.model_used, h)
            return Lesson(
                lesson_id  = memory.next_lesson_id(),
                episode_id = report.episode_id,
                action     = REINFORCE,
                trigger    = (
                    f"{n_consec} consecutive episodes with AUROC >= {config.AUROC_FLOOR} "
                    f"— current model tier is performing well"
                ),
                params     = {
                    "consecutive":       n_consec,
                    "reinforced_model":  used_tier,
                    "recent_aurocs":     [round(r.auroc, 4) for r in prior] + [round(report.auroc, 4)],
                },
                timestamp  = ts,
            )

        return None

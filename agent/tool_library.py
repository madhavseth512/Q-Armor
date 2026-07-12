"""Controlled tool library for the LLM Planner (Phase 9).

Defines the 6 allowed Reflexion actions as structured JSON schemas. The LLM
Planner must select exactly one tool per episode. The Verifier uses these
schemas to validate the LLM's choice before it is applied to policy.
"""

from __future__ import annotations

from agent.episodic_memory import (
    ALL_LESSON_ACTIONS,
    BINARY_ONLY,
    CALIBRATE_THRESHOLD,
    REINFORCE,
    SWITCH_MODEL,
    SWITCH_SUBSET,
    SWITCH_TYPE,
)

# ---------------------------------------------------------------------------
# Tool definitions (passed to LLM as structured context)
# ---------------------------------------------------------------------------

TOOLS: list[dict] = [
    {
        "name": REINFORCE,
        "description": (
            "The current model tier is performing well. Lock the current policy "
            "and reinforce the top-of-hierarchy model. Use when AUROC >= 0.70 "
            "for 3 or more consecutive episodes."
        ),
        "params": {},
        "when_to_use": "AUROC >= 0.70 sustained across recent episodes",
    },
    {
        "name": SWITCH_MODEL,
        "description": (
            "Demote the current quantum model tier one step down in the hierarchy "
            "[QSVM → VQC → QAE → CLASSICAL]. Use when AUROC < 0.70 and no "
            "distribution drift was detected — the kernel is over-specialised on "
            "the training distribution."
        ),
        "params": {
            "demote_from": "string — current tier name (e.g. 'QSVM')",
        },
        "when_to_use": "AUROC < 0.70 AND drift_detected=False",
    },
    {
        "name": SWITCH_SUBSET,
        "description": (
            "Re-select the k-means kernel subset from target-domain samples and "
            "retrain the quantum SVM. Use when AUROC < 0.70 AND drift has been "
            "confirmed by ADWIN — the input distribution has shifted."
        ),
        "params": {
            "pool_size": "int — number of target-domain samples to draw (default 300)",
        },
        "when_to_use": "AUROC < 0.70 AND drift_detected=True",
    },
    {
        "name": SWITCH_TYPE,
        "description": (
            "Trigger Stage 2 (RF typer) retraining on target-domain labelled samples. "
            "Use when cross-domain macro-F1 is near zero and SWITCH_SUBSET has already "
            "been applied to Stage 1 — Stage 2 also needs domain adaptation."
        ),
        "params": {
            "target_domain": "string — target dataset name (e.g. 'NF-UNSW-NB15')",
        },
        "when_to_use": "Stage 2 macro-F1 < 0.30 AND cross-domain deployment",
    },
    {
        "name": CALIBRATE_THRESHOLD,
        "description": (
            "Adjust the binary decision threshold up or down to fix a specific-class "
            "recall failure or FPR pathology. Use when a per-class F1 collapses to 0 "
            "even though Stage 1 AUROC is high — the threshold may be misplacing the "
            "decision boundary for that class."
        ),
        "params": {
            "direction": "string — 'lower' to increase recall, 'raise' to reduce FPR",
            "magnitude": "float — step size in [0.01, 0.15] (default 0.05)",
        },
        "when_to_use": "Per-class F1 = 0.000 AND S1 AUROC > 0.80, OR FPR@95 > 0.90",
    },
    {
        "name": BINARY_ONLY,
        "description": (
            "Disable Stage 2 multi-class typing and return binary detection only. "
            "Use when a multiclass model's macro-F1 collapses below 0.30, indicating "
            "the multi-class head has degenerated."
        ),
        "params": {},
        "when_to_use": "macro-F1 < 0.30 AND multiclass model active",
    },
]

# Quick lookup by name
TOOL_BY_NAME: dict[str, dict] = {t["name"]: t for t in TOOLS}


def tool_schema_str() -> str:
    """Return a compact human-readable tool menu for LLM prompts."""
    lines = ["Available tools (choose exactly one):"]
    for t in TOOLS:
        param_str = ", ".join(
            f"{k}: {v}" for k, v in t["params"].items()
        ) if t["params"] else "no params"
        lines.append(
            f"  • {t['name']}: {t['description']}\n"
            f"    params: {{{param_str}}}\n"
            f"    when: {t['when_to_use']}"
        )
    return "\n".join(lines)


def validate_action(action: str, params: dict) -> tuple[bool, str]:
    """Validate an LLM-chosen action + params against the tool library.

    Returns:
        (ok, error_message) — ok=True if valid, error_message="" if ok.
    """
    if action not in ALL_LESSON_ACTIONS:
        return False, f"Unknown action '{action}'. Must be one of {sorted(ALL_LESSON_ACTIONS)}."

    tool = TOOL_BY_NAME.get(action)
    if tool is None:
        return False, f"Action '{action}' has no tool definition."

    # Param type checks
    if action == SWITCH_SUBSET:
        pool = params.get("pool_size", 300)
        if not isinstance(pool, int) or pool < 10 or pool > 2000:
            return False, f"SWITCH_SUBSET pool_size must be int in [10, 2000], got {pool!r}."

    elif action == CALIBRATE_THRESHOLD:
        direction = params.get("direction", "lower")
        if direction not in ("lower", "raise"):
            return False, f"CALIBRATE_THRESHOLD direction must be 'lower' or 'raise', got {direction!r}."
        magnitude = params.get("magnitude", 0.05)
        try:
            magnitude = float(magnitude)
        except (TypeError, ValueError):
            return False, f"CALIBRATE_THRESHOLD magnitude must be a float, got {magnitude!r}."
        if not (0.01 <= magnitude <= 0.15):
            return False, f"CALIBRATE_THRESHOLD magnitude must be in [0.01, 0.15], got {magnitude}."

    return True, ""

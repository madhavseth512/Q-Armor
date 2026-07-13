"""LLM Planner — Phase 9 Reflexion component.

Takes the LLM Diagnoser's structured diagnosis and selects one action from
the controlled tool library. Uses Groq (Llama) with a strict JSON output format.

Output schema (JSON):
{
  "action":    "TOOL_NAME",
  "params":    {...},
  "rationale": "...",
  "confidence": 0.0-1.0
}
"""

from __future__ import annotations

import json
import os

from groq import Groq

from agent import agent_config as config
from agent.episodic_memory import (
    REINFORCE, SWITCH_MODEL, SWITCH_SUBSET,
    SWITCH_TYPE, CALIBRATE_THRESHOLD, BINARY_ONLY,
)
from agent.evaluator import EpisodeReport
from agent.tool_library import tool_schema_str
from reasoning.selector import TIER_TO_NAME

_CLIENT: Groq | None = None

_NAME_TO_TIER: dict[str, str] = {v: k for k, v in TIER_TO_NAME.items()}


def _client() -> Groq:
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = Groq(api_key=os.environ.get("GROQ_API_KEY", ""))
    return _CLIENT


_SYSTEM = """\
You are the Planner component of Q-Armor, an agentic quantum intrusion detection \
system. Given a diagnosis of the current episode and a tool library, select \
exactly one tool to apply next episode.

Respond ONLY with a valid JSON object:
{
  "action":     "<TOOL_NAME>",
  "params":     {},
  "rationale":  "<one or two sentences>",
  "confidence": <float 0.0-1.0>
}

Do not include text outside the JSON."""


def _build_prompt(
    report:    EpisodeReport,
    diagnosis: dict,
    policy:    dict,
    context_str: str,
) -> str:
    tier = _NAME_TO_TIER.get(report.model_used, "UNKNOWN")
    h    = policy.get("model_hierarchy", [])
    idx  = h.index(tier) if tier in h else 0
    next_tier = h[idx + 1] if idx + 1 < len(h) else h[-1]

    return f"""\
## Diagnosis

root_cause:   {diagnosis.get('root_cause')}
severity:     {diagnosis.get('severity')}
key_signals:  {diagnosis.get('key_signals')}
cross_domain: {diagnosis.get('cross_domain')}

## Episode Metrics

AUROC={report.auroc:.4f}  F1={report.binary_f1:.4f}  FPR@95={report.fpr_at_tpr95:.4f}
drift_detected={report.drift_detected}  model={report.model_used} (tier={tier})
Next tier in hierarchy if SWITCH_MODEL: {next_tier}
Current decision_threshold: {policy.get('decision_threshold', 0.5):.2f}

## Relevant History

{context_str}

## Tool Library

{tool_schema_str()}

Select the single best tool for this situation."""


def plan(
    report:      EpisodeReport,
    diagnosis:   dict,
    policy:      dict,
    context_str: str,
) -> dict:
    """Call Claude to choose an action from the tool library.

    Returns dict with keys: action, params, rationale, confidence.
    Falls back to heuristic plan if the API call fails.
    """
    prompt = _build_prompt(report, diagnosis, policy, context_str)
    try:
        response = _client().chat.completions.create(
            model      = config.LLM_MODEL,
            temperature= config.LLM_TEMPERATURE,
            max_tokens = config.LLM_MAX_TOKENS_DECISION,
            messages   = [
                {"role": "system", "content": _SYSTEM},
                {"role": "user",   "content": prompt},
            ],
        )
        result = json.loads(response.choices[0].message.content.strip())
        if result.get("params") is None:
            result["params"] = {}
        return result
    except Exception as exc:
        return _heuristic_plan(report, policy, str(exc))


def _heuristic_plan(report: EpisodeReport, policy: dict, error: str) -> dict:
    """Fallback plan when the LLM call fails — mirrors rule-based SelfReflector."""
    if report.auroc < config.AUROC_FLOOR and report.drift_detected:
        return {
            "action":     SWITCH_SUBSET,
            "params":     {"pool_size": config.SWITCH_SUBSET_N_CROSS},
            "rationale":  "Heuristic fallback: AUROC below floor with confirmed drift.",
            "confidence": 0.70,
            "_fallback":  True,
            "_error":     error,
        }
    elif report.auroc < config.AUROC_FLOOR:
        h = policy.get("model_hierarchy", [])
        tier = _NAME_TO_TIER.get(report.model_used, h[0] if h else "QSVM")
        return {
            "action":     SWITCH_MODEL,
            "params":     {"demote_from": tier},
            "rationale":  "Heuristic fallback: AUROC below floor, no drift.",
            "confidence": 0.65,
            "_fallback":  True,
            "_error":     error,
        }
    else:
        return {
            "action":     REINFORCE,
            "params":     {},
            "rationale":  "Heuristic fallback: episode appears healthy.",
            "confidence": 0.80,
            "_fallback":  True,
            "_error":     error,
        }

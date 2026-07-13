"""LLM Diagnoser — Phase 9 Reflexion component.

Takes an EpisodeReport + retrieved episodic context and calls Groq (Llama)
to produce a structured diagnosis of what went wrong and why. The diagnosis
is passed to the LLM Planner to select an action.

Output schema (JSON):
{
  "root_cause": "...",       -- concise root-cause hypothesis
  "severity":   "low|medium|high|critical",
  "key_signals": ["..."],    -- 2-4 bullet signals from the report
  "cross_domain": true/false -- whether domain shift is the primary factor
}
"""

from __future__ import annotations

import json
import os

from groq import Groq

from agent import agent_config as config
from agent.evaluator import EpisodeReport


_CLIENT: Groq | None = None


def _client() -> Groq:
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = Groq(api_key=os.environ.get("GROQ_API_KEY", ""))
    return _CLIENT


_SYSTEM = """\
You are the Diagnoser component of Q-Armor, an agentic quantum intrusion \
detection system. Your job is to analyse a single episode performance report \
and identify the root cause of any failure.

Respond ONLY with a valid JSON object matching this schema:
{
  "root_cause": "<one sentence>",
  "severity": "<low|medium|high|critical>",
  "key_signals": ["<signal 1>", "<signal 2>"],
  "cross_domain": <true|false>
}

Do not include any text outside the JSON object."""


def _build_prompt(report: EpisodeReport, context_str: str, policy: dict) -> str:
    return f"""\
## Current Episode Report (episode {report.episode_id})

- model_used:      {report.model_used}
- dataset:         {report.dataset}
- AUROC:           {report.auroc:.4f}  (floor: {config.AUROC_FLOOR})
- binary_F1:       {report.binary_f1:.4f}
- FPR@95:          {report.fpr_at_tpr95:.4f}
- AUPR:            {report.aupr:.4f}
- mean_confidence: {report.mean_confidence:.4f}
- drift_detected:  {report.drift_detected}
- n_samples:       {report.n_samples}

## Current Policy State

- model_hierarchy:    {policy.get('model_hierarchy')}
- decision_threshold: {policy.get('decision_threshold', 0.5):.2f}
- binary_only:        {policy.get('binary_only', False)}

## Relevant Past Episodes (RAG context)

{context_str}

Diagnose the root cause of this episode's performance."""


def diagnose(
    report:      EpisodeReport,
    context_str: str,
    policy:      dict,
) -> dict:
    """Call Claude to diagnose the episode.

    Returns a dict with keys: root_cause, severity, key_signals, cross_domain.
    Falls back to a heuristic diagnosis if the API call fails.
    """
    prompt = _build_prompt(report, context_str, policy)
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
        return json.loads(response.choices[0].message.content.strip())
    except Exception as exc:
        return _heuristic_diagnosis(report, str(exc))


def _heuristic_diagnosis(report: EpisodeReport, error: str) -> dict:
    """Fallback diagnosis when the LLM call fails."""
    if report.auroc < config.AUROC_FLOOR and report.drift_detected:
        cause = "Cross-domain distribution shift confirmed by ADWIN — kernel subset needs retraining on target domain."
        severity = "high"
        cross = True
    elif report.auroc < config.AUROC_FLOOR:
        cause = "AUROC below floor without drift — kernel may be over-specialised on training distribution."
        severity = "medium"
        cross = False
    elif report.fpr_at_tpr95 > 0.80:
        cause = "High false-positive rate despite acceptable AUROC — decision threshold may be too low."
        severity = "medium"
        cross = False
    else:
        cause = "Episode appears healthy."
        severity = "low"
        cross = False
    return {
        "root_cause":   cause,
        "severity":     severity,
        "key_signals":  [f"AUROC={report.auroc:.4f}", f"drift={report.drift_detected}", f"FPR95={report.fpr_at_tpr95:.4f}"],
        "cross_domain": cross,
        "_fallback":    True,
        "_error":       error,
    }

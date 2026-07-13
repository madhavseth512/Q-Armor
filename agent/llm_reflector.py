"""LLM Reflector — Phase 9 Reflexion component.

After the Planner selects an action, the Reflector writes a verbal lesson
into episodic memory. This lesson is stored alongside the structured Lesson
dataclass and is retrievable by SemanticMemory for future RAG context.

The verbal lesson is intentionally concise (1-2 sentences) to stay useful
as RAG context without overwhelming the LLM's context window.
"""

from __future__ import annotations

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
You are the Reflector component of Q-Armor. Write a single concise verbal \
lesson (1-2 sentences, ≤60 words) summarising what happened this episode and \
why the chosen action was taken. This lesson will be stored in episodic memory \
and retrieved in future episodes to guide adaptation decisions.

Write only the lesson text — no preamble, no labels."""


def _build_prompt(
    report:    EpisodeReport,
    action:    str,
    rationale: str,
    diagnosis: dict,
) -> str:
    return (
        f"Episode {report.episode_id}: "
        f"AUROC={report.auroc:.4f}, F1={report.binary_f1:.4f}, "
        f"FPR@95={report.fpr_at_tpr95:.4f}, drift={report.drift_detected}. "
        f"Diagnosis: {diagnosis.get('root_cause', 'unknown')}. "
        f"Action taken: {action}. Rationale: {rationale}. "
        f"Write the verbal lesson."
    )


def write_lesson(
    report:    EpisodeReport,
    action:    str,
    rationale: str,
    diagnosis: dict,
) -> str:
    """Call Gemini to write a verbal lesson for this episode.

    Returns the verbal lesson string. Returns a short fallback string on failure.
    """
    prompt = _build_prompt(report, action, rationale, diagnosis)
    try:
        response = _client().chat.completions.create(
            model      = config.LLM_MODEL,
            temperature= config.LLM_TEMPERATURE,
            max_tokens = config.LLM_MAX_TOKENS_REFLECTION,
            messages   = [
                {"role": "system", "content": _SYSTEM},
                {"role": "user",   "content": prompt},
            ],
        )
        return response.choices[0].message.content.strip()
    except Exception:
        return (
            f"Episode {report.episode_id}: AUROC={report.auroc:.4f}, "
            f"drift={report.drift_detected} → {action}."
        )

"""LLMReflexionAgent — Phase 9 drop-in replacement for SelfReflector.

Orchestrates the four-stage LLM pipeline:
  1. SemanticMemory.retrieve()  — RAG context from past episodes
  2. LLMDiagnoser.diagnose()   — structured root-cause analysis
  3. LLMPlanner.plan()         — action selection from tool library
  4. Verifier.verify()         — safety and logic guardrails
  5. LLMReflector.write_lesson() — verbal lesson for episodic memory

Falls back to the rule-based SelfReflector when:
  - ANTHROPIC_API_KEY is not set
  - An LLM call fails and the heuristic fallback produces an invalid action
  - The Verifier rejects the LLM decision

Public interface mirrors SelfReflector.reflect() so it is a drop-in swap
in any experiment that currently uses the rule-based reflector.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone

from agent import agent_config as config
from agent import llm_diagnoser, llm_planner, llm_reflector
from agent.episodic_memory import EpisodicMemory, Lesson
from agent.evaluator import EpisodeReport
from agent.reflector import SelfReflector
from agent.verifier import verify
from memory.semantic_memory import SemanticMemory


class LLMReflexionAgent:
    """Four-stage LLM Reflexion pipeline with verifier and rule-based fallback.

    Args:
        semantic_memory: Shared SemanticMemory instance. If None, a new one
                         is created from the default config path.
        verbose:         Print per-step diagnostics.
    """

    def __init__(
        self,
        semantic_memory: SemanticMemory | None = None,
        verbose: bool = True,
    ) -> None:
        self._sem_mem  = semantic_memory or SemanticMemory()
        self._fallback = SelfReflector()
        self._verbose  = verbose
        self._has_key  = bool(os.environ.get("ANTHROPIC_API_KEY", "").strip())

    # ------------------------------------------------------------------
    # Public API (mirrors SelfReflector.reflect)
    # ------------------------------------------------------------------

    def reflect(
        self,
        report: EpisodeReport,
        memory: EpisodicMemory,
        policy: dict,
    ) -> Lesson | None:
        """Run the LLM pipeline and return a Lesson (or None for healthy episodes).

        Falls back to the rule-based SelfReflector if the API key is absent or
        the LLM pipeline produces an invalid decision.
        """
        if not self._has_key:
            if self._verbose:
                print("  [LLMAgent] No ANTHROPIC_API_KEY — using rule-based fallback.")
            return self._fallback.reflect(report, memory, policy)

        ts = datetime.now(timezone.utc).isoformat()

        # ── 1. RAG retrieval ─────────────────────────────────────────────────
        query = (
            f"AUROC={report.auroc:.3f} drift={report.drift_detected} "
            f"model={report.model_used} dataset={report.dataset}"
        )
        context_entries = self._sem_mem.retrieve(query, top_k=config.LLM_RAG_TOP_K)
        context_str     = self._sem_mem.format_context(context_entries)
        if self._verbose:
            print(f"  [LLMAgent] RAG: retrieved {len(context_entries)} past episodes.")

        # ── 2. Diagnose ──────────────────────────────────────────────────────
        diagnosis = llm_diagnoser.diagnose(report, context_str, policy)
        if self._verbose:
            print(f"  [LLMAgent] Diagnosis: {diagnosis.get('root_cause', '?')} "
                  f"[{diagnosis.get('severity', '?')}]")

        # ── 3. Plan ──────────────────────────────────────────────────────────
        plan = llm_planner.plan(report, diagnosis, policy, context_str)
        if self._verbose:
            print(f"  [LLMAgent] Plan: action={plan.get('action')} "
                  f"confidence={plan.get('confidence', '?'):.2f}")

        # ── 4. Verify ────────────────────────────────────────────────────────
        vresult = verify(
            plan          = plan,
            policy        = policy,
            report_auroc  = report.auroc,
            report_drift  = report.drift_detected,
            report_model  = report.model_used,
        )

        if vresult.failed:
            if self._verbose:
                print(f"  [LLMAgent] Verifier rejected: {vresult.rejection_reason}")
                print("  [LLMAgent] Falling back to rule-based SelfReflector.")
            fallback_lesson = self._fallback.reflect(report, memory, policy)
            if fallback_lesson is not None:
                # Store in semantic memory even for fallback lessons
                self._sem_mem.store(
                    episode_id    = report.episode_id,
                    report_dict   = report.to_dict(),
                    action        = fallback_lesson.action,
                    verbal_lesson = f"[Fallback] {vresult.rejection_reason}",
                )
            return fallback_lesson

        action = vresult.action
        params = vresult.params

        # ── 5. Write verbal lesson ───────────────────────────────────────────
        verbal = llm_reflector.write_lesson(
            report    = report,
            action    = action,
            rationale = plan.get("rationale", ""),
            diagnosis = diagnosis,
        )
        if self._verbose:
            print(f"  [LLMAgent] Verbal lesson: {verbal[:80]}...")

        # Store in semantic memory before returning the Lesson
        self._sem_mem.store(
            episode_id    = report.episode_id,
            report_dict   = report.to_dict(),
            action        = action,
            verbal_lesson = verbal,
        )

        # ── Build Lesson ─────────────────────────────────────────────────────
        # Healthy episode check — if LLM chose REINFORCE and streak not met,
        # let rule-based handle it (REINFORCE needs consecutive-count logic).
        from agent.episodic_memory import REINFORCE
        if action == REINFORCE:
            rule_lesson = self._fallback.reflect(report, memory, policy)
            if rule_lesson is not None and rule_lesson.action == REINFORCE:
                rule_lesson.verbal_lesson = verbal
                rule_lesson.source = "llm"
                return rule_lesson
            elif rule_lesson is None:
                return None  # healthy, no lesson needed
            # LLM said REINFORCE but streak not met — use LLM action anyway
            # (the verifier already confirmed AUROC >= floor)

        return Lesson(
            lesson_id     = memory.next_lesson_id(),
            episode_id    = report.episode_id,
            action        = action,
            trigger       = plan.get("rationale", diagnosis.get("root_cause", "")),
            params        = params,
            timestamp     = ts,
            verbal_lesson = verbal,
            source        = "llm",
        )

    def quantum_diagnostics(self, report: EpisodeReport) -> dict:
        """Return quantum-specific signals from the EpisodeReport for LLM context.

        Extracts signals relevant to the quantum circuit (confidence, AUROC shape)
        that help the LLM distinguish quantum-specific from data-distribution issues.
        """
        return {
            "mean_confidence": report.mean_confidence,
            "auroc":           report.auroc,
            "fpr_at_tpr95":    report.fpr_at_tpr95,
            "low_confidence":  report.mean_confidence < config.CONFIDENCE_THRESHOLD,
            "auroc_gap":       round(report.auroc - config.AUROC_FLOOR, 4),
            "likely_quantum_issue": (
                report.mean_confidence < config.CONFIDENCE_THRESHOLD
                and report.auroc < config.AUROC_FLOOR
                and not report.drift_detected
            ),
        }

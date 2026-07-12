"""Semantic memory for Phase 9 LLM Reflexion agent.

Stores episode summaries and verbal lessons as searchable text. Retrieval
uses TF-IDF cosine similarity so no external embedding model is needed.

Two stores:
  - Episode summaries: structured text derived from each EpisodeReport + Lesson
  - Verbal lessons: free-text reflections written by LLMReflector

The top-K most similar past entries are retrieved and injected into the
LLM Diagnoser and Planner prompts as episodic context (RAG pattern).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from agent import agent_config as config


@dataclass
class MemoryEntry:
    episode_id:    int
    summary:       str   # structured text: metrics + action taken
    verbal_lesson: str   # LLM-generated narrative (empty for rule-based)
    action:        str   # lesson action taken
    auroc:         float
    drift:         bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def full_text(self) -> str:
        """Combined text used for TF-IDF retrieval."""
        parts = [self.summary]
        if self.verbal_lesson:
            parts.append(self.verbal_lesson)
        return " ".join(parts)


class SemanticMemory:
    """TF-IDF retrieval over episode summaries and verbal lessons.

    Entries are appended to a JSONL file and indexed in-memory. The index
    is rebuilt on load from the log file so memory survives restarts.
    """

    def __init__(self, path: str = config.SEMANTIC_MEMORY_PATH) -> None:
        self._path = path
        self._entries: list[MemoryEntry] = []
        self._vectorizer = TfidfVectorizer(max_features=500, stop_words="english")
        self._matrix: np.ndarray | None = None
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._load()

    # ------------------------------------------------------------------
    # Writing
    # ------------------------------------------------------------------

    def store(
        self,
        episode_id:    int,
        report_dict:   dict,
        action:        str,
        verbal_lesson: str = "",
    ) -> None:
        """Build an entry from an episode report dict and store it."""
        summary = self._summarise(episode_id, report_dict, action)
        entry = MemoryEntry(
            episode_id    = episode_id,
            summary       = summary,
            verbal_lesson = verbal_lesson,
            action        = action,
            auroc         = float(report_dict.get("auroc", 0.0)),
            drift         = bool(report_dict.get("drift_detected", False)),
        )
        self._entries.append(entry)
        with open(self._path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry.to_dict()) + "\n")
        self._rebuild_index()

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(self, query: str, top_k: int = config.LLM_RAG_TOP_K) -> list[MemoryEntry]:
        """Return the top-K most semantically similar past entries.

        Falls back to the most recent entries when the index is too small.
        """
        if not self._entries:
            return []
        if len(self._entries) <= top_k:
            return list(self._entries)
        if self._matrix is None or self._matrix.shape[0] == 0:
            return self._entries[-top_k:]

        try:
            q_vec = self._vectorizer.transform([query])
            sims = cosine_similarity(q_vec, self._matrix).flatten()
            top_idx = np.argsort(sims)[::-1][:top_k]
            return [self._entries[i] for i in top_idx]
        except Exception:
            return self._entries[-top_k:]

    def format_context(self, entries: list[MemoryEntry]) -> str:
        """Format retrieved entries as a readable context block for the LLM."""
        if not entries:
            return "No relevant past episodes found."
        lines = []
        for e in entries:
            lines.append(
                f"[Episode {e.episode_id}] AUROC={e.auroc:.4f} drift={e.drift} "
                f"action={e.action}\n  {e.summary}"
            )
            if e.verbal_lesson:
                lines.append(f"  Lesson: {e.verbal_lesson}")
        return "\n\n".join(lines)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _summarise(episode_id: int, report: dict, action: str) -> str:
        return (
            f"Episode {episode_id}: "
            f"AUROC={report.get('auroc', '?'):.4f} "
            f"F1={report.get('binary_f1', '?'):.4f} "
            f"FPR95={report.get('fpr_at_tpr95', '?'):.4f} "
            f"drift={report.get('drift_detected', '?')} "
            f"model={report.get('model_used', '?')} "
            f"-> action={action}"
        )

    def _rebuild_index(self) -> None:
        if len(self._entries) < 2:
            self._matrix = None
            return
        texts = [e.full_text() for e in self._entries]
        try:
            self._matrix = self._vectorizer.fit_transform(texts)
        except Exception:
            self._matrix = None

    def _load(self) -> None:
        if not os.path.exists(self._path):
            return
        with open(self._path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    self._entries.append(MemoryEntry(**d))
                except Exception:
                    continue
        if self._entries:
            self._rebuild_index()

    def __len__(self) -> int:
        return len(self._entries)

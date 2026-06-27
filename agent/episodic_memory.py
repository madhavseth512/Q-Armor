"""Episodic memory for the Reflexion outer loop.

Stores the full history of EpisodeReports and Lessons as an append-only JSONL
log, and maintains an in-memory policy that drives the inner loop's model
selection and subset re-selection behaviour.

Policy state fields:
    model_hierarchy   -- ordered list of tier names; the inner loop picks the
                         first tier whose model is registered and available.
    subset_reselect   -- True signals the inner loop to re-run k-means and
                         retrain all models before the next episode.
    binary_only       -- True forces Stage 1 (binary) only; skips multi-class.
    confidence_threshold -- sliding-window threshold for low-confidence trigger.
"""

from __future__ import annotations

import json
import os
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any

from agent import agent_config as config
from agent.evaluator import EpisodeReport


@dataclass
class Lesson:
    """A structured lesson written by the SelfReflector after one episode."""

    lesson_id:   int
    episode_id:  int
    action:      str   # SWITCH_SUBSET | SWITCH_MODEL | BINARY_ONLY | REINFORCE
    trigger:     str   # human-readable reason
    params:      dict  # action-specific parameters (e.g. demoted model name)
    timestamp:   str

    def to_dict(self) -> dict:
        return asdict(self)


# Sentinel when no lesson was written (healthy episode).
NO_LESSON = "NONE"

# Valid lesson action strings.
SWITCH_SUBSET = "SWITCH_SUBSET"
SWITCH_MODEL  = "SWITCH_MODEL"
BINARY_ONLY   = "BINARY_ONLY"
REINFORCE     = "REINFORCE"


class EpisodicMemory:
    """Append-only episode log + live policy for the Reflexion outer loop.

    Every episode produces one log entry: the EpisodeReport metrics plus the
    Lesson (or NO_LESSON). The policy is rebuilt from the log on load, so the
    agent can resume correctly after a restart.
    """

    def __init__(self, log_path: str = config.REFLEXION_LOG_PATH) -> None:
        self._log_path = log_path
        self._records: list[dict] = []
        self._lesson_counter = 0
        self._policy: dict[str, Any] = self._default_policy()
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

    # ------------------------------------------------------------------
    # Policy
    # ------------------------------------------------------------------

    @staticmethod
    def _default_policy() -> dict[str, Any]:
        return {
            "model_hierarchy":        list(config.MODEL_HIERARCHY),
            "subset_reselect":        False,
            "binary_only":            False,
            "confidence_threshold":   config.CONFIDENCE_THRESHOLD,
        }

    def get_policy(self) -> dict[str, Any]:
        """Return a copy of the current policy."""
        return deepcopy(self._policy)

    def _apply_lesson(self, lesson: Lesson) -> None:
        """Mutate the in-memory policy according to the lesson action."""
        h = self._policy["model_hierarchy"]

        if lesson.action == SWITCH_SUBSET:
            self._policy["subset_reselect"] = True

        elif lesson.action == SWITCH_MODEL:
            demoted = lesson.params.get("demoted_model")
            if demoted and demoted in h and h.index(demoted) < len(h) - 1:
                # Move the demoted tier one step down in the preference list.
                i = h.index(demoted)
                h[i], h[i + 1] = h[i + 1], h[i]

        elif lesson.action == BINARY_ONLY:
            self._policy["binary_only"] = True

        elif lesson.action == REINFORCE:
            # Promote the current top-of-hierarchy back to first position
            # (no-op if already first; restores order if SWITCH_MODEL swapped it).
            reinforced = lesson.params.get("reinforced_model")
            if reinforced and reinforced in h and h[0] != reinforced:
                h.remove(reinforced)
                h.insert(0, reinforced)

    def acknowledge_reselect(self) -> None:
        """Call after the inner loop has completed subset re-selection."""
        self._policy["subset_reselect"] = False

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record(self, report: EpisodeReport, lesson: Lesson | None) -> None:
        """Append an episode record (report + lesson) to memory and log.

        Args:
            report: The EpisodeReport produced by the Evaluator.
            lesson: The Lesson produced by the SelfReflector, or None if no
                    lesson was warranted (healthy episode).
        """
        entry: dict[str, Any] = {
            "record_type": "episode",
            "report":      report.to_dict(),
            "lesson":      lesson.to_dict() if lesson else NO_LESSON,
            "policy_after": self.get_policy(),
        }
        if lesson is not None:
            self._apply_lesson(lesson)
            entry["policy_after"] = self.get_policy()

        self._records.append(entry)
        with open(self._log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def recall(self, n: int) -> list[dict]:
        """Return the last n episode records (most recent last)."""
        return self._records[-n:] if n <= len(self._records) else list(self._records)

    def recent_reports(self, n: int) -> list[EpisodeReport]:
        """Reconstruct the last n EpisodeReports from stored records."""
        records = self.recall(n)
        reports = []
        for rec in records:
            r = rec["report"]
            reports.append(EpisodeReport(**r))
        return reports

    def next_lesson_id(self) -> int:
        self._lesson_counter += 1
        return self._lesson_counter

    def __len__(self) -> int:
        return len(self._records)

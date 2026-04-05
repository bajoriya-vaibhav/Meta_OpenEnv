"""
ChronoVeritas — EpisodeState: mutable runtime state for a single episode.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Literal, Optional, Set, Tuple

from env.models import DocMeta, MutationDecl, TimelineEntry

if TYPE_CHECKING:
    from env.models import Document

Phase = Literal["IDLE", "INITIALISED", "GRADING", "TERMINAL"]

# Valid one-way phase transitions
_VALID_TRANSITIONS: dict[Phase, set[Phase]] = {
    "IDLE":        {"INITIALISED"},
    "INITIALISED": {"GRADING", "TERMINAL"},
    "GRADING":     {"TERMINAL"},
    "TERMINAL":    set(),
}


@dataclass
class EpisodeState:
    # ── Identity ──────────────────────────────────────────────────────
    task_id: str = ""
    claim: str = ""
    difficulty: str = "easy"

    # ── Corpus ────────────────────────────────────────────────────────
    corpus: List[DocMeta] = field(default_factory=list)
    fetched_doc_ids: Set[str] = field(default_factory=set)

    # ── Agent-built structures ─────────────────────────────────────────
    agent_timeline: List[TimelineEntry] = field(default_factory=list)
    contradictions: List[Tuple[str, str]] = field(default_factory=list)
    declared_mutation: Optional[MutationDecl] = None

    # ── Budget tracking ────────────────────────────────────────────────
    current_step: int = 0
    max_steps: int = 15
    token_budget: int = 8_000
    token_used: int = 0

    # ── Lifecycle ─────────────────────────────────────────────────────
    phase: Phase = "IDLE"
    rewards_log: List[float] = field(default_factory=list)

    # ── Internal (not exposed via Observation) ─────────────────────────
    _fetched_docs_content: Dict[str, "Document"] = field(default_factory=dict)

    # ─────────────────────────────────────────────────────────────────
    # Computed properties
    # ─────────────────────────────────────────────────────────────────

    @property
    def token_budget_remaining(self) -> int:
        return max(self.token_budget - self.token_used, 0)

    @property
    def partial_reward_so_far(self) -> float:
        return sum(self.rewards_log)

    @property
    def steps_remaining(self) -> int:
        """How many step-costing actions are left before auto-termination."""
        return max(self.max_steps - self.current_step, 0)

    @property
    def is_done(self) -> bool:
        return self.phase == "TERMINAL"

    @property
    def budget_exhausted(self) -> bool:
        return self.current_step >= self.max_steps or self.token_budget_remaining == 0

    # ─────────────────────────────────────────────────────────────────
    # Phase management
    # ─────────────────────────────────────────────────────────────────

    def transition_phase(self, new_phase: Phase) -> None:
        """
        Validate and apply a phase transition.
        Raises ValueError on illegal transitions so callers never silently
        move backwards in the lifecycle.
        """
        allowed = _VALID_TRANSITIONS.get(self.phase, set())
        if new_phase not in allowed:
            raise ValueError(
                f"Illegal phase transition: {self.phase!r} → {new_phase!r}. "
                f"Allowed next phases: {allowed or {'<none — TERMINAL is final>'}}"
            )
        self.phase = new_phase

    # ─────────────────────────────────────────────────────────────────
    # Query helpers
    # ─────────────────────────────────────────────────────────────────

    def has_fetched(self, doc_id: str) -> bool:
        return doc_id in self.fetched_doc_ids

    def get_fetched_doc(self, doc_id: str) -> Optional["Document"]:
        return self._fetched_docs_content.get(doc_id)

    def corpus_doc_ids(self) -> Set[str]:
        """Set of doc_ids currently in the visible corpus metadata."""
        return {m.doc_id for m in self.corpus}

    def contradiction_pairs(self) -> List[Tuple[str, str]]:
        """Deduplicated contradiction pairs (order-normalised)."""
        seen: set[Tuple[str, str]] = set()
        result: List[Tuple[str, str]] = []
        for a, b in self.contradictions:
            key = (min(a, b), max(a, b))
            if key not in seen:
                seen.add(key)
                result.append((a, b))
        return result

    def timeline_for_doc(self, doc_id: str) -> List[TimelineEntry]:
        return [e for e in self.agent_timeline if e.doc_id == doc_id]

    # ─────────────────────────────────────────────────────────────────
    # Mutation helpers (guarded writes)
    # ─────────────────────────────────────────────────────────────────

    def consume_tokens(self, amount: int) -> bool:
        """
        Deduct *amount* tokens. Returns True on success, False if the
        budget would be exceeded (caller should abort the action).
        """
        if amount < 0:
            raise ValueError(f"Token amount must be non-negative, got {amount}")
        if self.token_budget_remaining < amount:
            return False
        self.token_used += amount
        return True

    def advance_step(self) -> None:
        """Increment the step counter (called by dispatcher for step-costing actions)."""
        self.current_step += 1

    def record_reward(self, delta: float) -> None:
        self.rewards_log.append(delta)

    def add_corpus_meta(self, meta: DocMeta) -> bool:
        """
        Add a DocMeta to the corpus if not already present.
        Returns True if it was newly added, False if already present.
        """
        if meta.doc_id in self.corpus_doc_ids():
            return False
        self.corpus.append(meta)
        return True

    def record_fetch(self, doc: "Document", token_cost: int) -> None:
        """Register a successfully fetched document."""
        self.fetched_doc_ids.add(doc.doc_id)
        self._fetched_docs_content[doc.doc_id] = doc
        self.token_used += token_cost

    # ─────────────────────────────────────────────────────────────────
    # Snapshot / restore (useful for beam-search agents or unit tests)
    # ─────────────────────────────────────────────────────────────────

    def snapshot(self) -> "EpisodeState":
        """Return a deep copy of the current state."""
        return copy.deepcopy(self)

    @classmethod
    def restore(cls, snap: "EpisodeState") -> "EpisodeState":
        """Return a deep copy of a previously taken snapshot."""
        return copy.deepcopy(snap)

    # ─────────────────────────────────────────────────────────────────
    # Debug
    # ─────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"EpisodeState("
            f"task={self.task_id!r}, "
            f"phase={self.phase!r}, "
            f"step={self.current_step}/{self.max_steps}, "
            f"tokens={self.token_used}/{self.token_budget}, "
            f"reward={self.partial_reward_so_far:.3f}, "
            f"fetched={len(self.fetched_doc_ids)}, "
            f"timeline={len(self.agent_timeline)}, "
            f"contradictions={len(self.contradictions)}"
            f")"
        )

    def summary_dict(self) -> dict:
        """Compact serialisable summary — handy for logging."""
        return {
            "task_id": self.task_id,
            "phase": self.phase,
            "difficulty": self.difficulty,
            "step": self.current_step,
            "max_steps": self.max_steps,
            "token_used": self.token_used,
            "token_budget": self.token_budget,
            "partial_reward": self.partial_reward_so_far,
            "fetched_count": len(self.fetched_doc_ids),
            "timeline_count": len(self.agent_timeline),
            "contradiction_count": len(self.contradictions),
            "mutation_declared": (
                self.declared_mutation.model_dump()
                if self.declared_mutation
                else None
            ),
        }
"""
ChronoVeritas — EpisodeState: mutable runtime state for a single episode.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Set, Tuple

from env.models import DocMeta, MutationDecl, TimelineEntry


@dataclass
class EpisodeState:
    task_id: str = ""
    claim: str = ""
    corpus: List[DocMeta] = field(default_factory=list)
    fetched_doc_ids: Set[str] = field(default_factory=set)
    agent_timeline: List[TimelineEntry] = field(default_factory=list)
    contradictions: List[Tuple[str, str]] = field(default_factory=list)
    declared_mutation: Optional[MutationDecl] = None
    current_step: int = 0
    max_steps: int = 15
    token_budget: int = 8000
    token_used: int = 0
    phase: Literal["IDLE", "INITIALISED", "GRADING", "TERMINAL"] = "IDLE"
    rewards_log: List[float] = field(default_factory=list)
    difficulty: str = "easy"
    # Internal bookkeeping
    _fetched_docs_content: Dict[str, "Document"] = field(default_factory=dict)  # type: ignore

    @property
    def token_budget_remaining(self) -> int:
        return max(self.token_budget - self.token_used, 0)

    @property
    def partial_reward_so_far(self) -> float:
        return sum(self.rewards_log)

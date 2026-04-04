"""
ChronoVeritas — BaseGrader: abstract grader with shared scoring helpers.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Set

from env.models import GradeResult, GroundTruth, TaskSpec, VerdictPayload
from env.state_manager import EpisodeState


def clip(val: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, val))


class BaseGrader(ABC):
    """Abstract base grader with deterministic sub-component helpers."""

    weights: Dict[str, float] = {}

    def __init__(self, task_spec: TaskSpec) -> None:
        self.gt: GroundTruth = task_spec.ground_truth
        self.task_spec = task_spec

    @abstractmethod
    def grade(self, state: EpisodeState, verdict: VerdictPayload) -> GradeResult:
        ...

    def grade_partial(self, state: EpisodeState) -> float:
        """Called on step-budget exhaustion. Awards half-credit for correct mutation."""
        score = 0.0
        if state.declared_mutation:
            if state.declared_mutation.doc_id == self.gt.gt_mutation_doc_id:
                score += self._w("mutation_point") * 0.5
        return score

    def _w(self, component: str) -> float:
        return self.weights.get(component, 0.0)

    # ── Shared scoring helpers ───────────────────────────────────────

    def _grade_verdict(self, agent_verdict: str) -> float:
        """1.0 if exact match, else 0.0."""
        return 1.0 if agent_verdict == self.gt.gt_verdict else 0.0

    def _grade_mutation_type(self, agent_mutation_type: str) -> float:
        """1.0 if exact match, else 0.0."""
        return 1.0 if agent_mutation_type == self.gt.gt_mutation_type else 0.0

    def _grade_mutation_point(self, agent_doc_id: Optional[str]) -> float:
        """1.0 if exact match, 0.5 if adjacent in gt timeline, else 0.0."""
        if agent_doc_id is None:
            return 0.0
        if agent_doc_id == self.gt.gt_mutation_doc_id:
            return 1.0

        # Check adjacency in gt_timeline
        gt_timeline = self.gt.gt_timeline
        if self.gt.gt_mutation_doc_id in gt_timeline and agent_doc_id in gt_timeline:
            gt_idx = gt_timeline.index(self.gt.gt_mutation_doc_id)
            agent_idx = gt_timeline.index(agent_doc_id)
            if abs(gt_idx - agent_idx) == 1:
                return 0.5

        return 0.0

    def _grade_provenance_f1(self, agent_chain: List[str]) -> float:
        """Set-based F1 between agent and ground-truth provenance chains."""
        pred_set = set(agent_chain)
        gt_set = set(self.gt.gt_provenance_chain)

        if not pred_set and not gt_set:
            return 1.0
        if not pred_set or not gt_set:
            return 0.0

        overlap = len(pred_set & gt_set)
        precision = overlap / len(pred_set)
        recall = overlap / len(gt_set)

        if precision + recall == 0.0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    def _grade_timeline(self, agent_timeline_ids: List[str]) -> float:
        """Kendall-tau normalized to [0, 1]."""
        gt_timeline_ids = self.gt.gt_timeline

        # Only docs in both
        common = [d for d in agent_timeline_ids if d in gt_timeline_ids]
        if len(common) < 2:
            return 0.0

        # Build rank maps
        agent_ranks = [agent_timeline_ids.index(d) for d in common]
        gt_ranks = [gt_timeline_ids.index(d) for d in common]

        try:
            from scipy.stats import kendalltau
            tau, _ = kendalltau(agent_ranks, gt_ranks)
            # Normalise from [-1, 1] to [0, 1]
            return clip((tau + 1.0) / 2.0)
        except Exception:
            return 0.0

    def _grade_hallucination(
        self,
        provenance_chain: List[str],
        fetched_doc_ids: Set[str],
        corpus_ids: Optional[List[str]] = None,
    ) -> float:
        """Penalty for citing unfetched or non-existent docs. Capped at 1.0."""
        if corpus_ids is None:
            corpus_ids = [d.doc_id for d in self.task_spec.corpus]
        corpus_set = set(corpus_ids)

        penalty = 0.0
        for doc_id in provenance_chain:
            if doc_id not in corpus_set:
                # Citing a non-existent doc
                penalty += 0.05
            elif doc_id not in fetched_doc_ids:
                # Citing a doc that was never fetched
                penalty += 0.05

        return min(penalty, 1.0)

"""
ChronoVeritas — BaseGrader: abstract grader with shared scoring helpers.

v2 changes:
  - Added source_reliability, early_detection, brier_penalty scoring
  - PENALTY_KEYS now includes "brier_penalty"
  - Provenance uses F1 only (no directional ordering)
  - All scoring is deterministic and reproducible
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections import Counter
from typing import Dict, List, Optional, Set

from env.models import GradeResult, GroundTruth, TaskSpec, VerdictPayload
from env.state_manager import EpisodeState

log = logging.getLogger(__name__)

# Tier-weight mapping for source reliability scoring
TIER_WEIGHTS: Dict[int, float] = {1: 1.0, 2: 0.6, 3: 0.2}


def clip(val: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """Clamp *val* to [lo, hi]."""
    return max(lo, min(hi, val))


class BaseGrader(ABC):
    """
    Abstract base grader with deterministic sub-component helpers.

    Subclasses must define a `weights` dict mapping component names to floats.
    Keys in PENALTY_KEYS are treated as penalty weights (subtracted), not
    positive score weights.  All other keys are positive contributions.

    Weight contract
    ---------------
    sum(w for k, w in weights.items() if k not in PENALTY_KEYS) should ≈ 1.0
    (warning logged if violated)
    """

    weights: Dict[str, float] = {}

    # Components that are *not* positive contributions
    PENALTY_KEYS: frozenset[str] = frozenset({"hallucination", "brier_penalty"})

    def __init__(self, task_spec: TaskSpec) -> None:
        self.gt: GroundTruth = task_spec.ground_truth
        self.task_spec = task_spec
        self._corpus_ids: List[str] = [d.doc_id for d in task_spec.corpus]
        self._corpus_id_set: Set[str] = set(self._corpus_ids)
        # Build doc_id → reliability_tier lookup from task corpus
        self._corpus_tiers: Dict[str, int] = {
            d.doc_id: d.reliability_tier for d in task_spec.corpus
        }
        self._validate_weights()

    # ── Weight validation ─────────────────────────────────────────────

    def _validate_weights(self) -> None:
        positive_total = sum(
            w for k, w in self.weights.items() if k not in self.PENALTY_KEYS
        )
        if self.weights and abs(positive_total - 1.0) > 0.06:
            log.warning(
                "%s: positive weights sum to %.4f (expected ~1.0). "
                "Scores may not be in [0, 1] as intended.",
                type(self).__name__,
                positive_total,
            )

    # ── Abstract interface ────────────────────────────────────────────

    @abstractmethod
    def grade(self, state: EpisodeState, verdict: VerdictPayload) -> GradeResult:
        """Full grading on episode completion."""
        ...

    # ── Partial grading (step-budget exhaustion) ──────────────────────

    def grade_partial(self, state: EpisodeState) -> float:
        """
        Called on step-budget exhaustion.
        Awards half-credit for a correct mutation point declaration, weighted
        by however much mutation_point contributes in this grader.
        Returns a value in [0, 0.5 * w("mutation_point")].
        """
        if not state.declared_mutation:
            return 0.0
        if state.declared_mutation.doc_id != self.gt.gt_mutation_doc_id:
            return 0.0
        return self._w("mutation_point") * 0.5

    # ── Weight helper ─────────────────────────────────────────────────

    def _w(self, component: str) -> float:
        """Return the weight for *component*, defaulting to 0.0."""
        return self.weights.get(component, 0.0)

    # ── Shared scoring helpers ────────────────────────────────────────

    def _grade_verdict(self, agent_verdict: str) -> float:
        """
        Returns 1.0 on exact match, 0.0 otherwise.
        Range: {0.0, 1.0}
        """
        return 1.0 if agent_verdict == self.gt.gt_verdict else 0.0

    def _grade_mutation_type(self, agent_mutation_type: str) -> float:
        """
        Returns 1.0 on exact match, 0.0 otherwise.
        Range: {0.0, 1.0}
        """
        return 1.0 if agent_mutation_type == self.gt.gt_mutation_type else 0.0

    def _grade_mutation_point(self, agent_doc_id: Optional[str]) -> float:
        """
        Returns:
          1.0 — exact match with gt_mutation_doc_id
          0.5 — adjacent (± 1 position) in gt_timeline
          0.0 — otherwise or None

        Range: {0.0, 0.5, 1.0}
        """
        if not agent_doc_id:
            return 0.0
        if agent_doc_id == self.gt.gt_mutation_doc_id:
            return 1.0

        gt_timeline = self.gt.gt_timeline
        gt_target = self.gt.gt_mutation_doc_id

        if not gt_target or gt_target not in gt_timeline or agent_doc_id not in gt_timeline:
            return 0.0

        gt_idx = gt_timeline.index(gt_target)
        agent_idx = gt_timeline.index(agent_doc_id)
        return 0.5 if abs(gt_idx - agent_idx) == 1 else 0.0

    def _grade_provenance_f1(self, agent_chain: List[str]) -> float:
        """
        Multiset-aware F1 between agent and ground-truth provenance chains.
        Handles duplicate doc_ids correctly (a doc cited twice counts twice).

        Range: [0.0, 1.0]
        """
        pred_counts = Counter(agent_chain)
        gt_counts = Counter(self.gt.gt_provenance_chain)

        if not pred_counts and not gt_counts:
            return 1.0
        if not pred_counts or not gt_counts:
            return 0.0

        # Multiset intersection
        overlap = sum(
            min(pred_counts[k], gt_counts[k]) for k in pred_counts if k in gt_counts
        )
        precision = overlap / sum(pred_counts.values())
        recall = overlap / sum(gt_counts.values())

        denom = precision + recall
        if denom == 0.0:
            return 0.0
        return 2.0 * precision * recall / denom

    def _grade_source_reliability(self, provenance_chain: List[str]) -> float:
        """
        Weighted average reliability tier of docs in agent's provenance chain.
        Tier 1 (official) = 1.0, Tier 2 (institutional) = 0.6, Tier 3 (informal) = 0.2.

        Rewards agents that anchor their chains to authoritative sources.

        Range: [0.0, 1.0]
        """
        if not provenance_chain:
            return 0.0

        weights = [
            TIER_WEIGHTS.get(self._corpus_tiers.get(doc_id, 2), 0.5)
            for doc_id in provenance_chain
            if doc_id in self._corpus_id_set
        ]
        if not weights:
            return 0.0
        return sum(weights) / len(weights)

    def _grade_timeline(self, agent_timeline_ids: List[str]) -> float:
        """
        Kendall-tau normalised to [0, 1] over documents present in both
        agent and ground-truth timelines.

        Falls back to a pure-Python concordant-pair count when scipy is
        unavailable (instead of silently returning 0.0).

        Range: [0.0, 1.0]
        """
        gt_timeline_ids = self.gt.gt_timeline

        if not gt_timeline_ids:
            return 1.0  # nothing to verify

        # Only score on docs present in both sequences
        gt_set = set(gt_timeline_ids)
        common = [d for d in agent_timeline_ids if d in gt_set]

        if len(common) < 2:
            # Award partial credit for having at least one correct doc
            return 0.25 if len(common) == 1 else 0.0

        agent_ranks = [agent_timeline_ids.index(d) for d in common]
        gt_ranks = [gt_timeline_ids.index(d) for d in common]

        try:
            from scipy.stats import kendalltau  # type: ignore[import]
            tau, _ = kendalltau(agent_ranks, gt_ranks)
            return clip((tau + 1.0) / 2.0)
        except Exception:
            # Pure-Python Kendall-tau (O(n²) — fine for short timelines)
            return self._kendall_tau_python(agent_ranks, gt_ranks)

    @staticmethod
    def _kendall_tau_python(a: List[int], b: List[int]) -> float:
        """
        Pure-Python Kendall-tau normalised to [0, 1].
        Returns 0.5 when concordant == discordant (no correlation).
        """
        n = len(a)
        concordant = discordant = 0
        for i in range(n):
            for j in range(i + 1, n):
                sign_a = a[j] - a[i]
                sign_b = b[j] - b[i]
                product = sign_a * sign_b
                if product > 0:
                    concordant += 1
                elif product < 0:
                    discordant += 1
        total = concordant + discordant
        if total == 0:
            return 0.5  # all ties — undefined, return neutral
        tau = (concordant - discordant) / total
        return clip((tau + 1.0) / 2.0)

    def _grade_efficiency(self, state: EpisodeState) -> float:
        """
        Step efficiency: fraction of budget *not* consumed.
        An agent that solves the task in fewer steps scores higher.

        Range: [0.0, 1.0]
        """
        if state.max_steps <= 0:
            return 0.0
        return clip(1.0 - state.current_step / state.max_steps)

    def _grade_early_detection(self, state: EpisodeState) -> float:
        """
        Binary bonus: 1.0 if the agent identified the correct mutation point
        within the first 40% of the step budget. 0.0 otherwise.

        This flag is set silently during set_mutation_point — the agent never
        sees it via reward signal during the episode.

        Range: {0.0, 1.0}
        """
        return 1.0 if state.early_detection_achieved else 0.0

    def _grade_brier_penalty(
        self, agent_confidence: float, agent_verdict: str
    ) -> float:
        """
        Brier-style penalty for miscalibrated confidence, with temporal decay
        based on provenance chain length.

        Penalises both overconfidence on wrong answers AND underconfidence
        on correct answers, while accounting for task complexity.

        Range: [0.0, 1.0]
        """
        chain_len = len(self.gt.gt_provenance_chain)
        decay = max(0.5, 1.0 - (chain_len - 2) * 0.1)

        correctness = 1.0 if agent_verdict == self.gt.gt_verdict else 0.0
        expected_conf = correctness * decay

        brier = (agent_confidence - expected_conf) ** 2
        return clip(brier)

    def _grade_hallucination(
        self,
        provenance_chain: List[str],
        fetched_doc_ids: Set[str],
        *,
        fabrication_penalty: float = 0.10,
        unread_penalty: float = 0.05,
    ) -> float:
        """
        Penalty for citing fabricated (not in corpus) or unread (not fetched) docs.

        Two-tier penalty:
          - fabrication_penalty per doc not in corpus   (more severe)
          - unread_penalty      per doc not fetched     (less severe)

        Total capped at 1.0 so it cannot exceed the weight it is multiplied by.

        Range: [0.0, 1.0]
        """
        penalty = 0.0
        for doc_id in provenance_chain:
            if doc_id not in self._corpus_id_set:
                penalty += fabrication_penalty
                log.debug("Hallucination: fabricated doc_id %r in provenance", doc_id)
            elif doc_id not in fetched_doc_ids:
                penalty += unread_penalty
                log.debug("Hallucination: unread doc_id %r cited in provenance", doc_id)
        return clip(penalty)

    # ── Debug helper ──────────────────────────────────────────────────

    def describe_weights(self) -> str:
        lines = [f"{type(self).__name__} weights:"]
        for k, w in self.weights.items():
            kind = "penalty" if k in self.PENALTY_KEYS else "score"
            lines.append(f"  {k:25s}  {w:.2f}  ({kind})")
        return "\n".join(lines)
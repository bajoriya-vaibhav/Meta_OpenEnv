"""
ChronoVeritas — UnifiedGrader: single grader for all difficulties.

The difficulty comes from the TASK (easy tasks are simpler to solve),
not from the grader.  This ensures training and evaluation use the
exact same scoring function — eliminating reward mismatch.

Components with missing ground-truth data (e.g. reconciliation on
easy tasks, timeline on tasks without gt_timeline) return 1.0,
meaning "nothing to verify → perfect score".
"""
from __future__ import annotations

from env.models import GradeResult, VerdictPayload
from env.state_manager import EpisodeState
from graders.base_grader import BaseGrader, clip


class UnifiedGrader(BaseGrader):
    """
    Fixed-weight grader used for ALL difficulty levels.

    Positive weights sum to 1.0.
    Penalty weights are subtracted separately.
    """

    weights = {
        # ── Positive components (sum = 1.00) ──────────────────────────
        "verdict":            0.25,
        "mutation_type":      0.15,
        "mutation_point":     0.15,
        "provenance":         0.15,
        "source_reliability": 0.08,
        "timeline":           0.05,
        "efficiency":         0.05,
        "early_detection":    0.05,
        "reconciliation":     0.07,
        # ── Penalties ─────────────────────────────────────────────────
        "hallucination":      0.12,
        "brier_penalty":      0.08,
    }

    def grade(self, state: EpisodeState, verdict: VerdictPayload) -> GradeResult:
        # ── Score every component ─────────────────────────────────────
        v   = self._grade_verdict(verdict.verdict)
        mt  = self._grade_mutation_type(verdict.mutation_type)
        mp  = self._grade_mutation_point(verdict.mutation_doc_id)
        pf1 = self._grade_provenance_f1(verdict.provenance_chain)
        sr  = self._grade_source_reliability(verdict.provenance_chain)

        agent_timeline_ids = [e.doc_id for e in state.agent_timeline]
        ts  = self._grade_timeline(agent_timeline_ids)

        eff = self._grade_efficiency(state)
        ed  = self._grade_early_detection(state)
        rec = self._grade_reconciliation(state)

        hp  = self._grade_hallucination(
            verdict.provenance_chain, state.fetched_doc_ids,
        )
        bp  = self._grade_brier_penalty(verdict.confidence, verdict.verdict)

        scores = {
            "verdict":            v,
            "mutation_type":      mt,
            "mutation_point":     mp,
            "provenance":         pf1,
            "source_reliability": sr,
            "timeline":           ts,
            "efficiency":         eff,
            "early_detection":    ed,
            "reconciliation":     rec,
        }

        total = (
            sum(self._w(k) * s for k, s in scores.items())
            - self._w("hallucination") * hp
            - self._w("brier_penalty") * bp
        )

        return GradeResult(
            total=clip(total),
            breakdown={
                **scores,
                "hallucination_penalty": hp,
                "brier_penalty":         bp,
                **{f"weighted_{k}": self._w(k) * s for k, s in scores.items()},
                "weighted_hallucination": self._w("hallucination") * hp,
                "weighted_brier":         self._w("brier_penalty") * bp,
            },
        )

    def grade_partial(self, state: EpisodeState) -> float:
        """
        On budget exhaustion: half-credit for mutation_point + partial
        provenance credit if the agent fetched relevant documents.
        """
        score = super().grade_partial(state)  # mutation_point half-credit

        # Partial provenance credit
        if state.declared_mutation and self.gt.gt_provenance_chain:
            fetched_correct = [
                d for d in state.fetched_doc_ids
                if d in self.gt.gt_provenance_chain
            ]
            if fetched_correct:
                recall = len(set(fetched_correct)) / len(set(self.gt.gt_provenance_chain))
                score += self._w("provenance") * recall * 0.5

        return clip(score)

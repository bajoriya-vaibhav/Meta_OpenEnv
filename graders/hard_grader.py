"""
ChronoVeritas — HardGrader (v2): all sub-components active.

Full spectrum including source reliability, enhanced provenance,
timeline ordering, early detection, reconciliation, and dual penalties.
"""
from __future__ import annotations

from env.models import GradeResult, VerdictPayload
from env.state_manager import EpisodeState
from graders.base_grader import BaseGrader, clip


class HardGrader(BaseGrader):
    weights = {
        "verdict":            0.20,
        "source_reliability": 0.08,
        "mutation_type":      0.10,
        "mutation_point":     0.10,
        "provenance":         0.15,
        "timeline":           0.07,
        "efficiency":         0.05,
        "early_detection":    0.05,
        "reconciliation":     0.15,   # hard-only: multi-source conflict coverage
        # penalties
        "hallucination":      0.15,
        "brier_penalty":      0.10,
    }

    def grade(self, state: EpisodeState, verdict: VerdictPayload) -> GradeResult:
        v   = self._grade_verdict(verdict.verdict)
        sr  = self._grade_source_reliability(verdict.provenance_chain)
        mt  = self._grade_mutation_type(verdict.mutation_type)
        mp  = self._grade_mutation_point(verdict.mutation_doc_id)
        pf1 = self._grade_provenance_f1(verdict.provenance_chain)

        agent_timeline_ids = [e.doc_id for e in state.agent_timeline]
        ts = self._grade_timeline(agent_timeline_ids)

        eff  = self._grade_efficiency(state)
        ed   = self._grade_early_detection(state)
        rec  = self._grade_reconciliation(state)
        hp   = self._grade_hallucination(
            verdict.provenance_chain, state.fetched_doc_ids,
        )
        bp   = self._grade_brier_penalty(verdict.confidence, verdict.verdict)

        scores = {
            "verdict":            v,
            "source_reliability": sr,
            "mutation_type":      mt,
            "mutation_point":     mp,
            "provenance":         pf1,
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
        On budget exhaustion: half-credit for mutation_point + partial provenance credit.
        """
        score = super().grade_partial(state)  # mutation_point half-credit

        # Partial provenance credit if agent cited at least one correct doc
        if state.declared_mutation and self.gt.gt_provenance_chain:
            fetched_correct = [
                d for d in state.fetched_doc_ids
                if d in self.gt.gt_provenance_chain
            ]
            if fetched_correct:
                recall = len(set(fetched_correct)) / len(set(self.gt.gt_provenance_chain))
                score += self._w("provenance") * recall * 0.5

        return clip(score)

    # ── HardGrader-specific helpers ───────────────────────────────────
    # NOTE: _grade_reconciliation is now in BaseGrader (returns 1.0
    # when gt_conflict_fields is empty so easy/medium tasks aren't penalised).
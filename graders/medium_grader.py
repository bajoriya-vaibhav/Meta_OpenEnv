"""
ChronoVeritas — MediumGrader (v2):
  verdict + source_reliability + mutation_type + mutation_point +
  provenance(F1) + efficiency + early_detection
  − hallucination − brier_penalty.
"""
from __future__ import annotations

from env.models import GradeResult, VerdictPayload
from env.state_manager import EpisodeState
from graders.base_grader import BaseGrader, clip


class MediumGrader(BaseGrader):
    weights = {
        "verdict":            0.25,
        "source_reliability": 0.10,
        "mutation_type":      0.20,
        "mutation_point":     0.20,
        "provenance":         0.10,
        "efficiency":         0.10,
        "early_detection":    0.05,
        # penalties
        "hallucination":      0.10,
        "brier_penalty":      0.08,
    }

    def grade(self, state: EpisodeState, verdict: VerdictPayload) -> GradeResult:
        v   = self._grade_verdict(verdict.verdict)
        sr  = self._grade_source_reliability(verdict.provenance_chain)
        mt  = self._grade_mutation_type(verdict.mutation_type)
        mp  = self._grade_mutation_point(verdict.mutation_doc_id)
        pf1 = self._grade_provenance_f1(verdict.provenance_chain)
        eff = self._grade_efficiency(state)
        ed  = self._grade_early_detection(state)
        hp  = self._grade_hallucination(
            verdict.provenance_chain, state.fetched_doc_ids,
        )
        bp  = self._grade_brier_penalty(verdict.confidence, verdict.verdict)

        scores = {
            "verdict":            v,
            "source_reliability": sr,
            "mutation_type":      mt,
            "mutation_point":     mp,
            "provenance":         pf1,
            "efficiency":         eff,
            "early_detection":    ed,
        }

        total = (
            sum(self._w(k) * s for k, s in scores.items())
            - self._w("hallucination")  * hp
            - self._w("brier_penalty")  * bp
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
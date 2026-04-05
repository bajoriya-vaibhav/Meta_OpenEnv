"""
ChronoVeritas — MediumGrader:
  verdict + mutation_type + mutation_point + efficiency − hallucination.

"""
from __future__ import annotations

from env.models import GradeResult, VerdictPayload
from env.state_manager import EpisodeState
from graders.base_grader import BaseGrader, clip


class MediumGrader(BaseGrader):
    weights = {
        "verdict":        0.35,
        "mutation_type":  0.25,
        "mutation_point": 0.30,
        "efficiency":     0.10,
        "hallucination":  0.10,   # penalty
    }

    def grade(self, state: EpisodeState, verdict: VerdictPayload) -> GradeResult:
        v   = self._grade_verdict(verdict.verdict)
        mt  = self._grade_mutation_type(verdict.mutation_type)
        mp  = self._grade_mutation_point(verdict.mutation_doc_id)
        eff = self._grade_efficiency(state)
        hp  = self._grade_hallucination(
            verdict.provenance_chain,
            state.fetched_doc_ids,
        )

        total = (
              self._w("verdict")        * v
            + self._w("mutation_type")  * mt
            + self._w("mutation_point") * mp
            + self._w("efficiency")     * eff
            - self._w("hallucination")  * hp
        )

        return GradeResult(
            total=clip(total),
            breakdown={
                # Raw sub-scores
                "verdict":               v,
                "mutation_type":         mt,
                "mutation_point":        mp,
                "efficiency":            eff,
                "hallucination_penalty": hp,
                # Weighted contributions
                "weighted_verdict":        self._w("verdict")        * v,
                "weighted_mutation_type":  self._w("mutation_type")  * mt,
                "weighted_mutation_point": self._w("mutation_point") * mp,
                "weighted_efficiency":     self._w("efficiency")     * eff,
                "weighted_hallucination":  self._w("hallucination")  * hp,
            },
        )
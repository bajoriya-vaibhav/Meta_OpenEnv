"""
ChronoVeritas — EasyGrader: verdict + mutation_point − hallucination.
"""
from __future__ import annotations

from env.models import GradeResult, VerdictPayload
from env.state_manager import EpisodeState
from graders.base_grader import BaseGrader, clip


class EasyGrader(BaseGrader):
    weights = {
        "verdict":        0.50,
        "mutation_point": 0.50,
        "hallucination":  0.10,   # penalty
    }

    def grade(self, state: EpisodeState, verdict: VerdictPayload) -> GradeResult:
        v  = self._grade_verdict(verdict.verdict)
        mp = self._grade_mutation_point(verdict.mutation_doc_id)
        hp = self._grade_hallucination(
            verdict.provenance_chain,
            state.fetched_doc_ids,
        )

        total = (
            self._w("verdict")        * v
            + self._w("mutation_point") * mp
            - self._w("hallucination")  * hp
        )

        return GradeResult(
            total=clip(total),
            breakdown={
                # Raw sub-scores
                "verdict":             v,
                "mutation_point":      mp,
                "hallucination_penalty": hp,
                # Weighted contributions (useful for debugging)
                "weighted_verdict":        self._w("verdict") * v,
                "weighted_mutation_point": self._w("mutation_point") * mp,
                "weighted_hallucination":  self._w("hallucination") * hp,
            },
        )
"""
ChronoVeritas — MediumGrader: verdict + mutation_type + mutation_point + efficiency − hallucination.
"""
from __future__ import annotations

from env.models import GradeResult, VerdictPayload
from env.state_manager import EpisodeState
from graders.base_grader import BaseGrader, clip


class MediumGrader(BaseGrader):
    weights = {
        "verdict": 0.35,
        "mutation_type": 0.25,
        "mutation_point": 0.30,
        "efficiency": 0.10,
        "hallucination": 0.10,
    }

    def grade(self, state: EpisodeState, verdict: VerdictPayload) -> GradeResult:
        v = self._grade_verdict(verdict.verdict)
        mt = self._grade_mutation_type(verdict.mutation_type)
        mp = self._grade_mutation_point(verdict.mutation_doc_id)
        eff = 1.0 - (state.current_step / state.max_steps) if state.max_steps > 0 else 0.0
        hp = self._grade_hallucination(
            verdict.provenance_chain, state.fetched_doc_ids
        )

        total = (
            self.weights["verdict"] * v
            + self.weights["mutation_type"] * mt
            + self.weights["mutation_point"] * mp
            + self.weights["efficiency"] * eff
            - self.weights["hallucination"] * hp
        )

        return GradeResult(
            total=clip(total),
            breakdown={
                "verdict": v,
                "mutation_type": mt,
                "mutation_point": mp,
                "efficiency": eff,
                "hallucination_penalty": hp,
            },
        )

"""
ChronoVeritas — HardGrader: all sub-components active.
"""
from __future__ import annotations

from env.models import GradeResult, VerdictPayload
from env.state_manager import EpisodeState
from graders.base_grader import BaseGrader, clip


class HardGrader(BaseGrader):
    weights = {
        "verdict": 0.25,
        "mutation_type": 0.20,
        "mutation_point": 0.20,
        "provenance": 0.15,
        "timeline": 0.10,
        "efficiency": 0.10,
        "hallucination": 0.15,
    }

    def grade(self, state: EpisodeState, verdict: VerdictPayload) -> GradeResult:
        v = self._grade_verdict(verdict.verdict)
        mt = self._grade_mutation_type(verdict.mutation_type)
        mp = self._grade_mutation_point(verdict.mutation_doc_id)
        pf1 = self._grade_provenance_f1(verdict.provenance_chain)

        # Timeline: extract doc_ids from agent_timeline entries
        agent_timeline_ids = [e.doc_id for e in state.agent_timeline]
        ts = self._grade_timeline(agent_timeline_ids)

        eff = 1.0 - (state.current_step / state.max_steps) if state.max_steps > 0 else 0.0
        hp = self._grade_hallucination(
            verdict.provenance_chain, state.fetched_doc_ids
        )

        scores = {
            "verdict": v,
            "mutation_type": mt,
            "mutation_point": mp,
            "provenance": pf1,
            "timeline": ts,
            "efficiency": eff,
        }

        total = sum(self.weights[k] * s for k, s in scores.items())
        total -= self.weights["hallucination"] * hp

        return GradeResult(
            total=clip(total),
            breakdown={**scores, "hallucination_penalty": hp},
        )

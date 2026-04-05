"""
ChronoVeritas — HardGrader: all sub-components active.

"""
from __future__ import annotations

from env.models import GradeResult, VerdictPayload
from env.state_manager import EpisodeState
from graders.base_grader import BaseGrader, clip


class HardGrader(BaseGrader):
    weights = {
        "verdict":        0.25,
        "mutation_type":  0.20,
        "mutation_point": 0.20,
        "provenance":     0.15,
        "timeline":       0.08,
        "efficiency":     0.06,   # step efficiency
        "token_eff":      0.06,   # token efficiency (NEW)
        "hallucination":  0.15,   # penalty
    }

    def grade(self, state: EpisodeState, verdict: VerdictPayload) -> GradeResult:
        v   = self._grade_verdict(verdict.verdict)
        mt  = self._grade_mutation_type(verdict.mutation_type)
        mp  = self._grade_mutation_point(verdict.mutation_doc_id)
        pf1 = self._grade_provenance_f1(verdict.provenance_chain)

        agent_timeline_ids = [e.doc_id for e in state.agent_timeline]
        ts = self._grade_timeline(agent_timeline_ids)

        eff      = self._grade_efficiency(state)
        tok_eff  = self._grade_token_efficiency(state)
        hp       = self._grade_hallucination(
            verdict.provenance_chain,
            state.fetched_doc_ids,
        )

        scores = {
            "verdict":       v,
            "mutation_type": mt,
            "mutation_point": mp,
            "provenance":    pf1,
            "timeline":      ts,
            "efficiency":    eff,
            "token_eff":     tok_eff,
        }

        total = (
            sum(self._w(k) * s for k, s in scores.items())
            - self._w("hallucination") * hp
        )

        return GradeResult(
            total=clip(total),
            breakdown={
                # Raw sub-scores
                **scores,
                "hallucination_penalty": hp,
                # Weighted contributions
                **{
                    f"weighted_{k}": self._w(k) * s
                    for k, s in scores.items()
                },
                "weighted_hallucination": self._w("hallucination") * hp,
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

    def _grade_token_efficiency(self, state: EpisodeState) -> float:
        """
        Fraction of token budget *not* consumed.
        Rewards agents that reach conclusions without reading everything.

        Range: [0.0, 1.0]
        """
        if state.token_budget <= 0:
            return 0.0
        return clip(1.0 - state.token_used / state.token_budget)
"""
ChronoVeritas — Main environment class orchestrating state, actions, and grading.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from env.actions import STEP_COSTING_ACTIONS, ActionDispatcher
from env.models import (
    Action,
    GradeResult,
    Observation,
    StepResult,
    TaskSpec,
    VerdictPayload,
)
from env.state_manager import EpisodeState
from graders.base_grader import BaseGrader
from graders.unified_grader import UnifiedGrader
from search.bm25_index import BM25Index
from search.corpus_store import CorpusStore

log = logging.getLogger(__name__)

TASKS_DIR = Path(__file__).resolve().parent.parent / "data" / "tasks"

# Default token budget per episode
DEFAULT_TOKEN_BUDGET: int = 8_000

# PBRS: scaling factor for intermediate potential-based shaping.
# Keeps intermediate signal at ~15% of total vs ~85% terminal reward.
SHAPING_SCALE: float = 0.15

# Tier-score mapping for authority sub-potential
_TIER_SCORES: dict[int, float] = {1: 1.0, 2: 0.5, 3: 0.1}


class ChronoVeritasEnv:
    """
    OpenEnv-compliant environment for claim lifecycle verification.

    Lifecycle
    ---------
    env = ChronoVeritasEnv()
    result = await env.reset(task_id="task_001")   # → StepResult (done=False)

    while not result.done:
        action = agent.act(result.observation)
        result = await env.step(action)

    await env.close()  # optional cleanup
    """

    def __init__(self, tasks_dir: Optional[Path] = None) -> None:
        self._tasks_dir = tasks_dir or TASKS_DIR
        self.tasks: List[TaskSpec] = []
        self._task_index: int = 0

        # Episode-scoped — reset on every reset()
        self.state: Optional[EpisodeState] = None
        self.dispatcher: Optional[ActionDispatcher] = None
        self.grader: Optional[BaseGrader] = None
        self._current_task: Optional[TaskSpec] = None

        # Persistent cross-episode infrastructure
        self.corpus_store = CorpusStore()
        self.bm25_index = BM25Index()

        self._load_tasks()

    # ── Task loading ──────────────────────────────────────────────────

    def _load_tasks(self) -> None:
        """
        Load task specs from tasks_dir/*.json.
        Malformed files are logged and skipped rather than crashing the env.
        """
        paths = sorted(self._tasks_dir.glob("*.json"))
        if not paths:
            log.warning("No task files found in %s", self._tasks_dir)

        for path in paths:
            try:
                with open(path) as f:
                    data = json.load(f)
                task = TaskSpec(**data)
                self.tasks.append(task)
                log.debug("Loaded task %r from %s", task.task_id, path.name)
            except Exception as exc:  # noqa: BLE001
                log.error("Failed to load task from %s: %s", path, exc)

    # ── Grader selection ──────────────────────────────────────────────

    def _select_grader(self, difficulty: str) -> BaseGrader:
        """Always return UnifiedGrader — difficulty comes from the task, not the grader."""
        assert self._current_task is not None, "No current task set"
        return UnifiedGrader(self._current_task)

    # ── Observation builder ───────────────────────────────────────────

    def _build_obs(self) -> Observation:
        """Construct an Observation snapshot from current episode state."""
        assert self.state is not None, "_build_obs() called before reset()"

        return Observation(
            claim=self.state.claim,
            corpus_metadata=list(self.state.corpus),
            retrieved_docs=list(self.state._fetched_docs_content.values()),
            agent_timeline=list(self.state.agent_timeline),
            flagged_contradictions=list(self.state.contradictions),
            current_step=self.state.current_step,
            max_steps=self.state.max_steps,
            token_budget_remaining=self.state.token_budget_remaining,
            partial_reward_so_far=self.state.partial_reward_so_far,
        )

    # ── Helpers ───────────────────────────────────────────────────────

    def _error_result(self, message: str, *, done: bool = False) -> StepResult:
        """Return a StepResult carrying only an error info dict."""
        assert self.state is not None
        return StepResult(
            observation=self._build_obs(),
            reward=0.0,
            done=done,
            info={"error": message},
        )

    # ── Public API ────────────────────────────────────────────────────

    async def reset(self, task_id: Optional[str] = None) -> StepResult:
        """
        Reset the environment for a new episode.

        Parameters
        ----------
        task_id:
            If supplied, the environment loads that specific task.
            If None, tasks are served round-robin.

        Returns
        -------
        StepResult with done=False and the initial observation.
        """
        if not self.tasks:
            raise RuntimeError(
                f"No tasks loaded from {self._tasks_dir}. "
                "Ensure the tasks directory contains valid JSON files."
            )

        if task_id is not None:
            task = next((t for t in self.tasks if t.task_id == task_id), None)
            if task is None:
                raise ValueError(
                    f"Unknown task_id: {task_id!r}. "
                    f"Available: {[t.task_id for t in self.tasks]}"
                )
        else:
            task = self.tasks[self._task_index % len(self.tasks)]
            self._task_index += 1

        self._current_task = task

        # Rebuild corpus & index for this task's documents
        self.corpus_store.load_from_task_corpus(task.corpus)
        self.bm25_index.build(task.corpus)

        # Select grader & dispatcher
        self.grader = self._select_grader(task.difficulty)
        self.dispatcher = ActionDispatcher(self.corpus_store, self.bm25_index)

        # Fresh episode state — agent starts blind, must discover docs via search
        self.state = EpisodeState(
            task_id=task.task_id,
            claim=task.claim,
            corpus=[],  # empty: agent must use search to discover documents
            max_steps=task.max_steps,
            token_budget=DEFAULT_TOKEN_BUDGET,
            phase="INITIALISED",
            difficulty=task.difficulty,
        )

        log.info(
            "Episode reset — task=%r difficulty=%r max_steps=%d corpus_size=%d",
            task.task_id,
            task.difficulty,
            task.max_steps,
            len(task.corpus),
        )

        return StepResult(
            observation=self._build_obs(),
            reward=0.0,
            done=False,
            info={"task_id": task.task_id, "difficulty": task.difficulty},
        )

    async def step(self, action: Action) -> StepResult:
        """
        Execute one action in the environment.

        Raises
        ------
        RuntimeError if called before reset().
        """
        if self.state is None or self.dispatcher is None or self.grader is None:
            raise RuntimeError("Call reset() before step()")

        if self.state.phase != "INITIALISED":
            return self._error_result(
                f"Episode is not active (phase={self.state.phase!r}). Call reset().",
                done=True,
            )

        # ── PBRS: compute potential BEFORE the action ─────────────────
        phi_before = self._compute_potential()

        # ── Dispatch the action ───────────────────────────────────────
        reward_delta, info, done = self.dispatcher.dispatch(
            action, self.state, self._current_task.ground_truth
        )

        # ── Advance step counter for step-costing actions ─────────────
        if action.type in STEP_COSTING_ACTIONS:
            self.state.advance_step()

        # ── PBRS: compute potential AFTER the action ──────────────────
        # Only apply shaping on non-terminal steps; terminal gets grader score
        if not done:
            phi_after = self._compute_potential()
            shaping_reward = SHAPING_SCALE * (phi_after - phi_before)
            reward_delta += shaping_reward
            info["shaping_reward"] = round(shaping_reward, 5)
            info["potential"] = round(phi_after, 4)

        # ── Auto-terminate on step budget exhaustion ──────────────────
        if not done and self.state.current_step >= self.state.max_steps:
            done = True
            partial = self.grader.grade_partial(self.state)
            reward_delta += partial
            info["auto_terminated"] = True
            info["partial_grade"] = partial
            log.info(
                "Episode auto-terminated — task=%r steps=%d partial_grade=%.3f",
                self.state.task_id,
                self.state.current_step,
                partial,
            )

        # ── Full grading on submit_verdict ────────────────────────────
        if action.type == "submit_verdict" and done:
            verdict_obj: Optional[VerdictPayload] = info.pop("_verdict_obj", None)
            if verdict_obj is None:
                # Fallback: reconstruct from info dict (should rarely happen)
                try:
                    verdict_obj = VerdictPayload(**info.get("verdict", {}))
                except Exception as exc:  # noqa: BLE001
                    log.error("Could not reconstruct VerdictPayload: %s", exc)

            if verdict_obj is not None:
                grade_result: GradeResult = self.grader.grade(self.state, verdict_obj)
                reward_delta += grade_result.total
                info["grade_breakdown"] = grade_result.breakdown
                info["final_score"] = grade_result.capped_total
                log.info(
                    "Episode graded — task=%r final_score=%.3f breakdown=%s",
                    self.state.task_id,
                    grade_result.capped_total,
                    grade_result.breakdown,
                )

        # ── Finalise state ────────────────────────────────────────────
        if done:
            try:
                self.state.transition_phase("TERMINAL")
            except ValueError:
                # Already TERMINAL or in an unexpected phase — log and continue
                log.warning(
                    "Unexpected phase %r when trying to set TERMINAL", self.state.phase
                )
                self.state.phase = "TERMINAL"

        self.state.record_reward(reward_delta)

        # Strip internal keys before returning
        public_info = {k: v for k, v in info.items() if not k.startswith("_")}

        return StepResult(
            observation=self._build_obs(),
            reward=reward_delta,
            done=done,
            info=public_info,
        )

    async def get_state(self) -> Observation:
        """Return the current observation without advancing the episode."""
        if self.state is None:
            raise RuntimeError("Call reset() first")
        return self._build_obs()

    def get_task_list(
        self, *, difficulty: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Return metadata for all available tasks.

        Parameters
        ----------
        difficulty:
            Optional filter — 'easy', 'medium', or 'hard'.
        """
        tasks = self.tasks
        if difficulty:
            tasks = [t for t in tasks if t.difficulty == difficulty]

        return [
            {
                "id": t.task_id,
                "difficulty": t.difficulty,
                "max_steps": t.max_steps,
                "claim": t.claim,
                "corpus_size": len(t.corpus),
            }
            for t in tasks
        ]

    def get_state_summary(self) -> Optional[Dict[str, Any]]:
        """
        Lightweight summary of the current episode — handy for dashboards
        and monitoring. Returns None if no episode is active.
        """
        if self.state is None:
            return None
        return self.state.summary_dict()

    async def close(self) -> None:
        """
        Release resources held by the environment.
        Safe to call multiple times.
        """
        self.state = None
        self.dispatcher = None
        self.grader = None
        self._current_task = None
        log.debug("ChronoVeritasEnv closed")

    # ── PBRS: Potential-Based Reward Shaping ─────────────────────────

    def _compute_potential(self) -> float:
        """
        Deterministic forensic progress estimator.

        Measures investigation quality from observable state only — never
        accesses ground truth.  Returns a scalar in [0.0, 1.0].

        Five sub-potentials:
          1. Exploration — fraction of total corpus actually fetched
          2. Authority   — average reliability tier of fetched docs
          3. Contradiction density — evidence-based conflict flags
          4. Hypothesis narrowing  — grounding of declared mutation point
          5. Evidence coherence    — timeline/contradiction grounding
        """
        assert self.state is not None and self._current_task is not None
        state = self.state
        total_corpus = len(self._current_task.corpus)
        if total_corpus == 0:
            return 0.0

        fetched = len(state.fetched_doc_ids)

        # 1. Exploration: fraction of total corpus read
        exploration = min(fetched / total_corpus, 1.0) if total_corpus > 0 else 0.0

        # 2. Authority: average tier quality of fetched docs
        authority = 0.0
        if fetched > 0:
            corpus_tiers = {d.doc_id: d.reliability_tier for d in self._current_task.corpus}
            tier_vals = [
                _TIER_SCORES.get(corpus_tiers.get(did, 3), 0.1)
                for did in state.fetched_doc_ids
            ]
            authority = sum(tier_vals) / len(tier_vals)

        # 3. Contradiction density: flags relative to possible pairs
        contradiction = 0.0
        if fetched >= 2:
            max_pairs = fetched * (fetched - 1) / 2
            actual = len(state.contradiction_pairs())
            contradiction = min(actual / max(max_pairs, 1) * 3.0, 1.0)

        # 4. Hypothesis narrowing: is declared mutation backed by evidence?
        narrowing = 0.0
        if state.declared_mutation is not None:
            did = state.declared_mutation.doc_id
            read_it = did in state.fetched_doc_ids
            flagged_it = any(did in (a, b) for a, b in state.contradictions)
            timeline_it = any(e.doc_id == did for e in state.agent_timeline)
            narrowing = (float(read_it) + float(flagged_it) + float(timeline_it)) / 3.0

        # 5. Evidence coherence: grounded timeline + contradiction entries
        coherence = 0.0
        if state.agent_timeline:
            grounded = sum(1 for e in state.agent_timeline if e.doc_id in state.fetched_doc_ids)
            coherence += 0.5 * grounded / len(state.agent_timeline)
        if state.contradictions:
            both_read = sum(
                1 for a, b in state.contradictions
                if a in state.fetched_doc_ids and b in state.fetched_doc_ids
            )
            coherence += 0.5 * both_read / len(state.contradictions)

        # Weighted sum (weights sum to 1.0)
        phi = (
            0.25 * exploration
            + 0.15 * authority
            + 0.20 * contradiction
            + 0.20 * narrowing
            + 0.20 * coherence
        )
        return phi
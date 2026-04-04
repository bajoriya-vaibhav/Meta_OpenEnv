"""
ChronoVeritas — Main environment class orchestrating state, actions, and grading.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from env.actions import STEP_COSTING_ACTIONS, ActionDispatcher
from env.models import (
    Action,
    DocMeta,
    Document,
    Observation,
    StepResult,
    TaskSpec,
    VerdictPayload,
)
from env.state_manager import EpisodeState
from graders.base_grader import BaseGrader
from graders.easy_grader import EasyGrader
from graders.hard_grader import HardGrader
from graders.medium_grader import MediumGrader
from search.bm25_index import BM25Index
from search.corpus_store import CorpusStore


TASKS_DIR = Path(__file__).resolve().parent.parent / "data" / "tasks"


class ChronoVeritasEnv:
    """OpenEnv-compliant environment for claim lifecycle verification."""

    def __init__(self) -> None:
        self.tasks: List[TaskSpec] = []
        self._task_index: int = 0
        self.state: Optional[EpisodeState] = None
        self.corpus_store = CorpusStore()
        self.bm25_index = BM25Index()
        self.dispatcher: Optional[ActionDispatcher] = None
        self.grader: Optional[BaseGrader] = None
        self._current_task: Optional[TaskSpec] = None

        # Load all tasks
        self._load_tasks()

    def _load_tasks(self) -> None:
        """Load task specs from data/tasks/*.json."""
        for path in sorted(TASKS_DIR.glob("*.json")):
            with open(path) as f:
                data = json.load(f)
            self.tasks.append(TaskSpec(**data))

    def _select_grader(self, difficulty: str) -> BaseGrader:
        assert self._current_task is not None
        if difficulty == "easy":
            return EasyGrader(self._current_task)
        elif difficulty == "medium":
            return MediumGrader(self._current_task)
        else:
            return HardGrader(self._current_task)

    def _build_obs(self) -> Observation:
        """Build the current observation from episode state."""
        assert self.state is not None

        retrieved_docs = list(self.state._fetched_docs_content.values())

        return Observation(
            claim=self.state.claim,
            corpus_metadata=list(self.state.corpus),
            retrieved_docs=retrieved_docs,
            agent_timeline=list(self.state.agent_timeline),
            flagged_contradictions=list(self.state.contradictions),
            current_step=self.state.current_step,
            max_steps=self.state.max_steps,
            token_budget_remaining=self.state.token_budget_remaining,
            partial_reward_so_far=self.state.partial_reward_so_far,
        )

    # ── Public API ───────────────────────────────────────────────────

    async def reset(self, task_id: Optional[str] = None) -> StepResult:
        """Reset env with a specific task or round-robin."""
        if task_id:
            task = next((t for t in self.tasks if t.task_id == task_id), None)
            if task is None:
                raise ValueError(f"Unknown task_id: {task_id}")
        else:
            task = self.tasks[self._task_index % len(self.tasks)]
            self._task_index += 1

        self._current_task = task

        # Build corpus & index
        self.corpus_store.load_from_task_corpus(task.corpus)
        self.bm25_index.build(task.corpus)

        # Select grader
        self.grader = self._select_grader(task.difficulty)

        # Init dispatcher
        self.dispatcher = ActionDispatcher(self.corpus_store, self.bm25_index)

        # Init episode state
        self.state = EpisodeState(
            task_id=task.task_id,
            claim=task.claim,
            corpus=self.corpus_store.all_metas(),
            max_steps=task.max_steps,
            token_budget=8000,
            phase="INITIALISED",
            difficulty=task.difficulty,
        )

        return StepResult(
            observation=self._build_obs(),
            reward=0.0,
            done=False,
            info={"task_id": task.task_id, "difficulty": task.difficulty},
        )

    async def step(self, action: Action) -> StepResult:
        """Execute one action in the environment."""
        assert self.state is not None
        assert self.dispatcher is not None
        assert self.grader is not None
        assert self._current_task is not None

        # Guard: must be INITIALISED
        if self.state.phase != "INITIALISED":
            return StepResult(
                observation=self._build_obs(),
                reward=0.0,
                done=True,
                info={"error": "Call reset() before step(). Current phase: " + self.state.phase},
            )

        # Dispatch
        reward_delta, info, done = self.dispatcher.dispatch(
            action, self.state, self._current_task.ground_truth
        )

        # Advance step counter for step-costing actions
        if action.type in STEP_COSTING_ACTIONS:
            self.state.current_step += 1

        # Auto-terminate on step budget exhaustion
        if self.state.current_step >= self.state.max_steps and not done:
            done = True
            partial = self.grader.grade_partial(self.state)
            reward_delta += partial
            info["auto_terminated"] = True
            info["partial_grade"] = partial

        # If submit_verdict triggered, run full grader
        if action.type == "submit_verdict" and done:
            verdict_data = info.get("_verdict_obj")
            if verdict_data is None:
                # Reconstruct from info
                vd = info.get("verdict", {})
                verdict_data = VerdictPayload(**vd)

            grade_result = self.grader.grade(self.state, verdict_data)
            # Add any partial reward already accumulated (e.g. from set_mutation_point)
            reward_delta += grade_result.total
            info["grade_breakdown"] = grade_result.breakdown
            info["final_score"] = grade_result.total
            # Remove internal object
            info.pop("_verdict_obj", None)

        # Update phase
        if done:
            self.state.phase = "TERMINAL"

        # Record reward
        self.state.rewards_log.append(reward_delta)

        return StepResult(
            observation=self._build_obs(),
            reward=reward_delta,
            done=done,
            info={k: v for k, v in info.items() if not k.startswith("_")},
        )

    async def get_state(self) -> Observation:
        """Return the current observation."""
        return self._build_obs()

    def get_task_list(self) -> List[Dict[str, Any]]:
        """Return list of available tasks."""
        return [
            {
                "id": t.task_id,
                "difficulty": t.difficulty,
                "max_steps": t.max_steps,
                "claim": t.claim,
            }
            for t in self.tasks
        ]

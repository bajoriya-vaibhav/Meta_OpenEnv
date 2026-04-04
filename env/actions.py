"""
ChronoVeritas — ActionDispatcher: routes all 6 action types.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

from env.models import (
    Action,
    DocMeta,
    Document,
    MutationDecl,
    TimelineEntry,
    VerdictPayload,
)
from env.state_manager import EpisodeState
from search.bm25_index import BM25Index
from search.corpus_store import CorpusStore

# Actions that consume a step
STEP_COSTING_ACTIONS = {"search", "fetch_doc"}


class ActionDispatcher:
    """Dispatch actions against the episode state."""

    def __init__(self, corpus_store: CorpusStore, bm25_index: BM25Index) -> None:
        self.corpus = corpus_store
        self.index = bm25_index

    def dispatch(
        self,
        action: Action,
        state: EpisodeState,
        ground_truth: Any = None,
    ) -> Tuple[float, Dict[str, Any], bool]:
        """
        Execute an action and return (reward_delta, info, done).
        """
        handler = getattr(self, f"_handle_{action.type}", None)
        if handler is None:
            return 0.0, {"error": f"Unknown action type: {action.type}"}, False
        return handler(action.payload, state, ground_truth)

    # ── search ───────────────────────────────────────────────────────

    def _handle_search(
        self, payload: Dict[str, Any], state: EpisodeState, gt: Any
    ) -> Tuple[float, Dict[str, Any], bool]:
        query = payload.get("query", "")
        date_from = payload.get("date_from")
        date_to = payload.get("date_to")

        results = self.index.query(query, date_from=date_from, date_to=date_to, top_k=10)

        # Merge results into corpus metadata (avoid duplicates)
        existing_ids = {m.doc_id for m in state.corpus}
        for meta in results:
            if meta.doc_id not in existing_ids:
                state.corpus.append(meta)
                existing_ids.add(meta.doc_id)

        return 0.0, {"search_results": [m.model_dump() for m in results]}, False

    # ── fetch_doc ────────────────────────────────────────────────────

    def _handle_fetch_doc(
        self, payload: Dict[str, Any], state: EpisodeState, gt: Any
    ) -> Tuple[float, Dict[str, Any], bool]:
        doc_id = payload.get("doc_id", "")
        doc = self.corpus.get_doc(doc_id)

        if doc is None:
            return 0.0, {"error": f"Document {doc_id} not found"}, False

        # Check token budget BEFORE consuming
        token_cost = len(doc.content) // 4
        if state.token_budget_remaining <= 0:
            return 0.0, {"error": "token_budget_exceeded"}, False

        # Record fetch
        state.fetched_doc_ids.add(doc_id)
        state._fetched_docs_content[doc_id] = doc
        state.token_used += token_cost

        return 0.0, {"fetched": doc_id, "token_cost": token_cost}, False

    # ── add_timeline_event ───────────────────────────────────────────

    def _handle_add_timeline_event(
        self, payload: Dict[str, Any], state: EpisodeState, gt: Any
    ) -> Tuple[float, Dict[str, Any], bool]:
        doc_id = payload.get("doc_id", "")
        event_label = payload.get("event_label", "")
        timestamp = payload.get("timestamp")

        entry = TimelineEntry(doc_id=doc_id, event_label=event_label, timestamp=timestamp)
        state.agent_timeline.append(entry)

        return 0.0, {"timeline_added": doc_id}, False

    # ── flag_contradiction ───────────────────────────────────────────

    def _handle_flag_contradiction(
        self, payload: Dict[str, Any], state: EpisodeState, gt: Any
    ) -> Tuple[float, Dict[str, Any], bool]:
        doc_id_a = payload.get("doc_id_a", "")
        doc_id_b = payload.get("doc_id_b", "")

        state.contradictions.append((doc_id_a, doc_id_b))

        return 0.0, {"contradiction_flagged": [doc_id_a, doc_id_b]}, False

    # ── set_mutation_point ───────────────────────────────────────────

    def _handle_set_mutation_point(
        self, payload: Dict[str, Any], state: EpisodeState, gt: Any
    ) -> Tuple[float, Dict[str, Any], bool]:
        doc_id = payload.get("doc_id", "")
        mutation_type = payload.get("mutation_type", "none")

        state.declared_mutation = MutationDecl(doc_id=doc_id, mutation_type=mutation_type)

        # Partial reward
        reward = 0.0
        if gt is not None:
            if doc_id == gt.gt_mutation_doc_id:
                reward += 0.20
            if mutation_type == gt.gt_mutation_type:
                reward += 0.20

        return reward, {"mutation_declared": doc_id, "partial_reward": reward}, False

    # ── submit_verdict ───────────────────────────────────────────────

    def _handle_submit_verdict(
        self, payload: Dict[str, Any], state: EpisodeState, gt: Any
    ) -> Tuple[float, Dict[str, Any], bool]:
        # Parse the verdict payload
        verdict = VerdictPayload(
            verdict=payload.get("verdict", "unverifiable"),
            mutation_type=payload.get("mutation_type", "none"),
            mutation_doc_id=payload.get("mutation_doc_id"),
            provenance_chain=payload.get("provenance_chain", []),
            confidence=payload.get("confidence", 0.5),
        )

        state.phase = "GRADING"

        # Store verdict in info for grader to pick up later
        return 0.0, {"verdict": verdict.model_dump(), "_verdict_obj": verdict}, True

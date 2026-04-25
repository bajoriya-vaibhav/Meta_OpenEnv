"""
ChronoVeritas — ActionDispatcher: routes all 6 action types.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, Tuple

from pydantic import ValidationError

from env.models import (
    Action,
    MutationDecl,
    TimelineEntry,
    VerdictPayload,
)
from env.state_manager import EpisodeState
from search.bm25_index import BM25Index
from search.corpus_store import CorpusStore

log = logging.getLogger(__name__)

# Actions that consume a step
STEP_COSTING_ACTIONS: frozenset[str] = frozenset({"search", "fetch_doc"})

# Stop words for label grounding computation
_STOP_WORDS: frozenset[str] = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "in", "of", "and", "or",
    "that", "it", "its", "by", "for", "to", "at", "this", "with", "after",
    "on", "from", "as", "be", "has", "had", "have", "not", "but", "no",
})

# Result type: (reward_delta, info_dict, done)
DispatchResult = Tuple[float, Dict[str, Any], bool]

def _ok(info: Dict[str, Any], *, reward: float = 0.0, done: bool = False) -> DispatchResult:
    return reward, info, done


def _err(message: str, *, reward: float = 0.0, done: bool = False) -> DispatchResult:
    log.debug("ActionDispatcher error: %s", message)
    return reward, {"error": message}, done


class ActionDispatcher:
    """Dispatch actions against the episode state."""

    def __init__(self, corpus_store: CorpusStore, bm25_index: BM25Index) -> None:
        self.corpus = corpus_store
        self.index = bm25_index

    # ── Entry point ───────────────────────────────────────────────────

    def dispatch(
        self,
        action: Action,
        state: EpisodeState,
        ground_truth: Any = None,
    ) -> DispatchResult:
        """
        Execute an action and return (reward_delta, info, done).

        Never raises — all exceptions are caught and returned as error info.
        """
        handler = getattr(self, f"_handle_{action.type}", None)
        if handler is None:
            return _err(f"Unknown action type: {action.type!r}")
        try:
            return handler(action.payload, state, ground_truth)
        except Exception as exc:  # noqa: BLE001
            log.exception("Unhandled exception in handler for %r", action.type)
            return _err(f"Internal error in {action.type!r}: {exc}")

    # ── search ────────────────────────────────────────────────────────

    def _handle_search(
        self, payload: Dict[str, Any], state: EpisodeState, gt: Any
    ) -> DispatchResult:
        query: str = payload.get("query", "").strip()
        if not query:
            return _err("search.query must be a non-empty string")

        date_from: int | None = payload.get("date_from")
        date_to: int | None = payload.get("date_to")

        if date_from is not None and date_to is not None and date_from > date_to:
            return _err("search.date_from must be <= date_to")

        results = self.index.query(query, date_from=date_from, date_to=date_to, top_k=10)

        newly_added: list[str] = []
        for meta in results:
            if state.add_corpus_meta(meta):
                newly_added.append(meta.doc_id)

        return _ok(
            {
                "search_results": [m.model_dump() for m in results],
                "newly_added_to_corpus": newly_added,
                "total_results": len(results),
            }
        )

    # ── fetch_doc ─────────────────────────────────────────────────────

    def _handle_fetch_doc(
        self, payload: Dict[str, Any], state: EpisodeState, gt: Any
    ) -> DispatchResult:
        doc_id: str = payload.get("doc_id", "").strip()
        if not doc_id:
            return _err("fetch_doc.doc_id must be a non-empty string")

        # Agent must discover a document via search before fetching it
        if doc_id not in state.corpus_doc_ids():
            return _err(
                f"fetch_doc: Document {doc_id!r} has not been discovered yet. "
                "Use the 'search' action first to find documents."
            )

        # Idempotent: warn but succeed if already fetched
        if state.has_fetched(doc_id):
            doc = state.get_fetched_doc(doc_id)
            return _ok(
                {
                    "fetched": doc_id,
                    "token_cost": 0,
                    "note": "already fetched — no token cost",
                    "cached": True,
                    "estimated_tokens": doc.estimated_tokens if doc else 0,
                }
            )

        doc = self.corpus.get_doc(doc_id)
        if doc is None:
            return _err(f"fetch_doc: Document {doc_id!r} not found in corpus store")

        token_cost = doc.estimated_tokens

        if state.token_budget_remaining == 0:
            return _err(
                "fetch_doc: token budget exhausted — cannot fetch more documents",
                done=False,
            )

        # Partial fetch when token_cost exceeds remaining budget
        actual_cost = min(token_cost, state.token_budget_remaining)
        consumed = state.consume_tokens(actual_cost)
        if not consumed:
            return _err("fetch_doc: token budget exhausted (race condition guard)")

        state.record_fetch(doc, token_cost=0)  # cost already applied above

        return _ok(
            {
                "fetched": doc_id,
                "token_cost": actual_cost,
                "estimated_tokens": token_cost,
                "truncated": actual_cost < token_cost,
                "cached": False,
            }
        )

    # ── add_timeline_event ────────────────────────────────────────────

    def _handle_add_timeline_event(
        self, payload: Dict[str, Any], state: EpisodeState, gt: Any
    ) -> DispatchResult:
        doc_id: str = payload.get("doc_id", "").strip()
        event_label: str = payload.get("event_label", "").strip()
        timestamp: int | None = payload.get("timestamp")

        if not doc_id:
            return _err("add_timeline_event.doc_id must be non-empty")
        if not event_label:
            return _err("add_timeline_event.event_label must be non-empty")

        # Soft-warn if doc hasn't been fetched yet
        if not state.has_fetched(doc_id) and doc_id not in state.corpus_doc_ids():
            log.warning(
                "add_timeline_event: doc_id %r not in fetched or corpus; adding anyway", doc_id
            )

        # Deduplicate: same (doc_id, event_label, timestamp) triple
        duplicate = any(
            e.doc_id == doc_id and e.event_label == event_label and e.timestamp == timestamp
            for e in state.agent_timeline
        )
        if duplicate:
            return _ok(
                {
                    "timeline_added": False,
                    "doc_id": doc_id,
                    "note": "duplicate entry — skipped",
                }
            )

        entry = TimelineEntry(doc_id=doc_id, event_label=event_label, timestamp=timestamp)
        state.agent_timeline.append(entry)

        # No ad-hoc bonus — intermediate reward is handled by PBRS in environment.py
        return _ok(
            {
                "timeline_added": True,
                "doc_id": doc_id,
                "total_events": len(state.agent_timeline),
            },
        )

    # ── flag_contradiction ────────────────────────────────────────────

    def _handle_flag_contradiction(
        self, payload: Dict[str, Any], state: EpisodeState, gt: Any
    ) -> DispatchResult:
        doc_id_a: str = payload.get("doc_id_a", "").strip()
        doc_id_b: str = payload.get("doc_id_b", "").strip()

        if not doc_id_a or not doc_id_b:
            return _err("flag_contradiction: both doc_id_a and doc_id_b must be non-empty")

        if doc_id_a == doc_id_b:
            return _err("flag_contradiction: doc_id_a and doc_id_b must be different documents")

        # Normalised deduplication
        normalised = (min(doc_id_a, doc_id_b), max(doc_id_a, doc_id_b))
        existing = {(min(a, b), max(a, b)) for a, b in state.contradictions}
        if normalised in existing:
            return _ok(
                {
                    "contradiction_flagged": False,
                    "pair": [doc_id_a, doc_id_b],
                    "note": "duplicate contradiction — skipped",
                }
            )

        state.contradictions.append((doc_id_a, doc_id_b))

        both_fetched = state.has_fetched(doc_id_a) and state.has_fetched(doc_id_b)

        # No ad-hoc bonus — intermediate reward is handled by PBRS in environment.py
        return _ok(
            {
                "contradiction_flagged": True,
                "pair": [doc_id_a, doc_id_b],
                "total_contradictions": len(state.contradictions),
                "both_docs_fetched": both_fetched,
            },
        )

    # ── set_mutation_point ────────────────────────────────────────────

    def _handle_set_mutation_point(
        self, payload: Dict[str, Any], state: EpisodeState, gt: Any
    ) -> DispatchResult:
        doc_id: str = payload.get("doc_id", "").strip()
        mutation_type: str = payload.get("mutation_type", "none")

        if not doc_id:
            return _err("set_mutation_point.doc_id must be non-empty")

        valid_types = {"distortion", "omission", "fabrication", "context_shift", "none"}
        if mutation_type not in valid_types:
            return _err(
                f"set_mutation_point.mutation_type {mutation_type!r} is invalid; "
                f"must be one of {sorted(valid_types)}"
            )

        state.declared_mutation = MutationDecl(doc_id=doc_id, mutation_type=mutation_type)  # type: ignore[arg-type]

        # ── Evidence grounding (NO GT comparison — no answer leakage) ─
        read_it = doc_id in state.fetched_doc_ids
        flagged_it = any(doc_id in (a, b) for a, b in state.contradictions)
        timeline_it = any(e.doc_id == doc_id for e in state.agent_timeline)

        grounding = (float(read_it) + float(flagged_it) + float(timeline_it)) / 3.0

        # ── Silent early detection check (stored, NOT returned as reward) ─
        if gt is not None:
            if (doc_id == gt.gt_mutation_doc_id
                    and mutation_type == gt.gt_mutation_type):
                steps_used_pct = state.current_step / max(state.max_steps, 1)
                if steps_used_pct <= 0.40:
                    state.early_detection_achieved = True
                    state.early_detection_step_pct = steps_used_pct

        # Reward is ALWAYS 0.0 — no answer leakage
        return _ok(
            {
                "mutation_declared": doc_id,
                "mutation_type": mutation_type,
                "evidence_grounding": round(grounding, 3),
                "evidence_details": {
                    "document_fetched": read_it,
                    "contradiction_flagged": flagged_it,
                    "timeline_annotated": timeline_it,
                },
            },
            reward=0.0,
        )

    # ── submit_verdict ────────────────────────────────────────────────

    def _handle_submit_verdict(
        self, payload: Dict[str, Any], state: EpisodeState, gt: Any
    ) -> DispatchResult:
        # Validate the payload strictly via Pydantic
        try:
            verdict = VerdictPayload(
                verdict=payload.get("verdict", "unverifiable"),
                mutation_type=payload.get("mutation_type", "none"),
                mutation_doc_id=payload.get("mutation_doc_id"),
                provenance_chain=payload.get("provenance_chain", []),
                confidence=payload.get("confidence", 0.5),
            )
        except ValidationError as exc:
            # Surface validation errors clearly rather than silently accepting bad data
            return _err(f"submit_verdict: invalid payload — {exc}", done=True)

        state.transition_phase("GRADING")

        return (
            0.0,
            {
                "verdict": verdict.model_dump(),
                "_verdict_obj": verdict,  # consumed by ChronoVeritasEnv.step()
            },
            True,
        )
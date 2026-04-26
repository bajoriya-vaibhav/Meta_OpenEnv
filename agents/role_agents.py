"""
ChronoVeritas — Multi-Agent Role Agents

Defines the three cooperating agent roles for the fact-checking pipeline:
  RetrieverAgent: BM25 corpus search + document fetching
  AnalystAgent:   Cross-document contradiction detection + timeline building
  ArbiterAgent:   Evidence synthesis + final verdict (the GRPO-trained role)

These are used by:
  - MultiAgentController (this module) for orchestrated rollouts
  - train_grpo.py heuristic_retriever/heuristic_analyst for training prompts
  - test_pretrain_validation.py for validation rollouts
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Literal, Optional

from env.models import Action, AgentRole


# ── Role-based action permissions ─────────────────────────────────────────

ROLE_PERMISSIONS: Dict[str, set] = {
    "retriever": {"search", "fetch_doc", "send_message", "request_evidence"},
    "analyst":   {"flag_contradiction", "add_timeline_event", "update_hypothesis",
                  "send_message", "request_evidence"},
    "arbiter":   {"submit_verdict", "set_mutation_point", "send_message",
                  "request_evidence", "update_hypothesis"},
}


def is_action_permitted(role: str, action_type: str) -> bool:
    """Check if a role is allowed to take a given action."""
    perms = ROLE_PERMISSIONS.get(role, set())
    return action_type in perms


# ── Retriever Agent ───────────────────────────────────────────────────────

class RetrieverAgent:
    """
    Heuristic Retriever: searches corpus via BM25 and fetches relevant docs.
    No LLM needed — pure algorithmic search.
    """

    def __init__(self, top_k: int = 6):
        self.top_k = top_k

    def generate_actions(self, observation: Dict, claim: str) -> List[Dict]:
        """
        Given an observation, produce retriever actions:
          1. search(claim)
          2. fetch_doc(doc_id) for top results
        """
        actions = []

        # Always start with a search
        actions.append({"type": "search", "query": claim[:200]})

        # Fetch docs found in corpus metadata
        corpus_meta = observation.get("corpus_metadata", [])
        fetched_ids = {d.get("doc_id") for d in observation.get("retrieved_docs", [])
                       if isinstance(d, dict)}

        for doc in corpus_meta[:self.top_k]:
            if isinstance(doc, dict) and doc.get("doc_id") not in fetched_ids:
                actions.append({"type": "fetch_doc", "doc_id": doc["doc_id"]})

        return actions


# ── Analyst Agent ─────────────────────────────────────────────────────────

class AnalystAgent:
    """
    Heuristic Analyst: compares documents to find number/date mismatches
    and builds a timeline. No LLM needed — pure string analysis.
    """

    @staticmethod
    def _extract_numbers(text: str) -> List[str]:
        """Extract numeric values (including $, %, decimals) from text."""
        return re.findall(r'[\$]?\d+[\.,]?\d*[%BMKTbmkt]*', text)

    def generate_actions(self, observation: Dict, claim: str) -> List[Dict]:
        """
        Analyse fetched documents and produce:
          - flag_contradiction actions for number mismatches
          - add_timeline_event actions for dated events
        """
        actions = []
        docs = observation.get("retrieved_docs", [])
        docs = [d for d in docs if isinstance(d, dict)]

        if not docs:
            return actions

        # Extract numbers from each doc
        doc_numbers: Dict[str, List[str]] = {}
        for doc in docs:
            doc_id = doc.get("doc_id", "?")
            content = doc.get("content", "")
            doc_numbers[doc_id] = self._extract_numbers(content)

        # Flag contradictions between doc pairs
        doc_ids = list(doc_numbers.keys())
        for i in range(len(doc_ids)):
            for j in range(i + 1, len(doc_ids)):
                id_a, id_b = doc_ids[i], doc_ids[j]
                nums_a = set(doc_numbers[id_a])
                nums_b = set(doc_numbers[id_b])
                if nums_a and nums_b and nums_a != nums_b:
                    diff = nums_a.symmetric_difference(nums_b)
                    if diff:
                        actions.append({
                            "type": "flag_contradiction",
                            "doc_id_a": id_a,
                            "doc_id_b": id_b,
                            "description": f"Number mismatch: {diff}",
                        })

        # Add timeline events
        for doc in docs:
            doc_id = doc.get("doc_id", "?")
            date_str = doc.get("date_str", "")
            title = doc.get("title", "")
            timestamp = doc.get("timestamp", 0)
            if title:
                actions.append({
                    "type": "add_timeline_event",
                    "doc_id": doc_id,
                    "event_label": title[:80],
                    "timestamp": timestamp if isinstance(timestamp, int) else 0,
                })

        return actions[:10]  # cap to avoid flooding


# ── Arbiter Agent ─────────────────────────────────────────────────────────

class ArbiterAgent:
    """
    The Arbiter is the GRPO-trained decision maker.

    In training: the prompt is built by heuristic_retriever + heuristic_analyst,
                 and the Arbiter's response is the JSON verdict.
    In inference: uses the trained LoRA model to generate the verdict.
    In testing:  uses an external LLM (Groq) as a stand-in.
    """

    @staticmethod
    def build_context(observation: Dict, claim: str,
                      contradictions: List[str],
                      timeline: List[str]) -> str:
        """Build the Arbiter's decision context from team findings."""
        parts = [f"CLAIM: {claim}\n"]

        # Documents
        docs = observation.get("retrieved_docs", [])
        if docs:
            parts.append("=== RETRIEVER FINDINGS ===")
            for doc in docs:
                if not isinstance(doc, dict):
                    continue
                parts.append(
                    f"  [{doc.get('doc_id','?')}] Tier {doc.get('reliability_tier', 2)} "
                    f"| {doc.get('title','?')}"
                )
                content = doc.get("content", "")[:400]
                parts.append(f"  {content}\n")

        # Analyst findings
        parts.append("=== ANALYST FINDINGS ===")
        if contradictions:
            parts.append("Contradictions:")
            for c in contradictions:
                parts.append(f"  - {c}")
        else:
            parts.append("  (no contradictions detected)")

        if timeline:
            parts.append("Timeline:")
            for t in timeline:
                parts.append(f"  {t}")

        return "\n".join(parts)


# ── Multi-Agent Controller ────────────────────────────────────────────────

class MultiAgentController:
    """
    Orchestrates the three-role cooperative workflow:
      1. Retriever searches and fetches documents
      2. Analyst flags contradictions and builds timeline
      3. Arbiter reviews findings and submits verdict

    Used by test_pretrain_validation.py and can be used for inference.
    """

    ROLE_SEQUENCE = ["retriever", "analyst", "arbiter"]

    def __init__(self):
        self.retriever = RetrieverAgent()
        self.analyst = AnalystAgent()
        self.arbiter = ArbiterAgent()

    def get_role_agent(self, role: str):
        """Return the agent for a given role."""
        return {
            "retriever": self.retriever,
            "analyst": self.analyst,
            "arbiter": self.arbiter,
        }.get(role)

    def generate_actions_for_role(
        self, role: str, observation: Dict, claim: str
    ) -> List[Dict]:
        """Generate heuristic actions for a given role."""
        if role == "retriever":
            return self.retriever.generate_actions(observation, claim)
        elif role == "analyst":
            return self.analyst.generate_actions(observation, claim)
        else:
            return []  # Arbiter actions come from LLM, not heuristics

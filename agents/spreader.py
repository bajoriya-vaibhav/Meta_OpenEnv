"""
ChronoVeritas v2 — Spreader Agent (rule-based, deterministic)

The Spreader takes a MutationResult and constructs a full TaskSpec JSON
that is compatible with the v1 environment.

Corpus structure is governed by difficulty:
  - easy:   3 docs, 0 noise
  - medium: 6 docs, 2 noise
  - hard:   12 docs, 4 noise
"""
from __future__ import annotations

import random
import time
from typing import Any, Dict, List, Optional
from uuid import uuid4

from agents.mutator import MutationResult
from agents.task_bank import (
    NOISE_TEMPLATES,
    SEED_FACTS,
    TIER2_NEWS_TEMPLATES,
    TIER3_BLOG_TEMPLATES,
    SeedFact,
)


# ── Noise configuration by difficulty ─────────────────────────────────────
NOISE_CONFIG = {
    "easy":   {"n_docs": 3,  "n_noise": 0, "noise_tiers": []},
    "medium": {"n_docs": 6,  "n_noise": 2, "noise_tiers": [2, 3]},
    "hard":   {"n_docs": 12, "n_noise": 4, "noise_tiers": [1, 2, 2, 3]},
}

# Max steps per difficulty
MAX_STEPS = {"easy": 20, "medium": 25, "hard": 35}

# Unrelated entities/orgs for noise documents
NOISE_ENTITIES = [
    "Pinnacle Industries", "Cascade Biotech", "SummitView Capital",
    "Horizon Energy Group", "Clearwater Analytics", "Atlas Manufacturing",
    "Vertex Solutions", "Ironclad Financial", "NovaTech Systems",
    "Pacific Minerals Corp",
]

NOISE_ORGS = [
    "National Research Foundation", "International Trade Council",
    "Regional Development Authority", "Public Health Alliance",
]

NOISE_DOMAINS = [
    "renewable energy", "digital infrastructure", "workforce development",
    "supply chain management", "environmental compliance",
]


class Spreader:
    """
    Rule-based task construction engine.

    Usage:
        spreader = Spreader(seed=42)
        task = spreader.spread(mutation_result, difficulty="easy")
        # task is a dict ready for the v1 environment
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        self.rng = random.Random(seed)
        self._doc_counter = 0

    # ── Public API ──────────────────────────────────────────────────────

    def spread(
        self,
        mutation: MutationResult,
        difficulty: str = "easy",
    ) -> Dict[str, Any]:
        """
        Build a full TaskSpec dict from a MutationResult.

        Returns a dict compatible with env.models.TaskSpec.
        """
        if difficulty not in NOISE_CONFIG:
            raise ValueError(f"Unknown difficulty: {difficulty!r}")

        self._doc_counter = 0  # Reset per task
        corpus = self._build_corpus(mutation, difficulty)
        gt_timeline = self._build_timeline(corpus)
        gt_mutation_doc_id = self._find_mutation_doc(corpus)
        gt_provenance_chain = self._build_provenance(corpus, gt_mutation_doc_id)

        task_id = f"{difficulty.upper()[:3]}-{uuid4().hex[:8]}"

        return {
            "task_id": task_id,
            "difficulty": difficulty,
            "max_steps": MAX_STEPS[difficulty],
            "claim": mutation.false_claim,
            "ground_truth": {
                "gt_verdict": "false",  # All generated tasks are "false" (mutated)
                "gt_mutation_type": mutation.mutation_type,
                "gt_mutation_doc_id": gt_mutation_doc_id,
                "gt_provenance_chain": gt_provenance_chain,
                "gt_timeline": gt_timeline,
                "gt_conflict_fields": self._find_conflicts(corpus, difficulty),
                "corpus_ids": [d["doc_id"] for d in corpus],
            },
            "corpus": corpus,
        }

    # ── Corpus construction ────────────────────────────────────────────

    def _build_corpus(
        self,
        result: MutationResult,
        difficulty: str,
    ) -> List[Dict[str, Any]]:
        """Build the document corpus based on difficulty level."""
        corpus: List[Dict[str, Any]] = []
        fact = result.seed_fact

        if difficulty == "easy":
            # 3 docs: Tier-1 true source, Tier-2 corroborating, Tier-2/3 mutated
            corpus.append(self._make_tier1_doc(fact))
            corpus.append(self._make_tier2_corroborating(fact))
            corpus.append(self._make_mutated_doc(result, tier=2))

        elif difficulty == "medium":
            # 6 docs total (4 real + 2 noise)
            corpus.append(self._make_tier1_doc(fact))
            corpus.append(self._make_tier2_corroborating(fact))
            corpus.append(self._make_tier3_mutation_origin(result))
            corpus.append(self._make_tier2_propagation(result))
            # 2 noise docs
            corpus += self._make_noise_docs(result, n=2, tiers=[2, 3])

        elif difficulty == "hard":
            # 12 docs total (8 real + 4 noise)
            corpus.append(self._make_tier1_doc(fact))
            corpus.append(self._make_tier1_conflicting(fact))
            corpus.append(self._make_tier2_corroborating(fact))
            corpus.append(self._make_tier3_mutation_origin(result))
            corpus += self._make_tier2_propagation_chain(result, n=2)
            corpus += self._make_tier3_amplifiers(result, n=2)
            # 4 noise docs
            corpus += self._make_noise_docs(result, n=4, tiers=[1, 2, 2, 3])

        # Shuffle to prevent positional bias
        self.rng.shuffle(corpus)
        return corpus

    # ── Document factories ─────────────────────────────────────────────

    def _next_doc_id(self) -> str:
        self._doc_counter += 1
        return f"DOC-{self._doc_counter:04d}"

    def _base_timestamp(self, fact: SeedFact, offset_days: int = 0) -> int:
        """Generate a plausible Unix timestamp based on the fact's date."""
        # Parse year-month-day from fact.true_date
        parts = fact.true_date.split("-")
        try:
            import datetime
            base = datetime.datetime(
                int(parts[0]), int(parts[1]), int(parts[2]),
                tzinfo=datetime.timezone.utc
            )
        except (ValueError, IndexError):
            base = datetime.datetime(2024, 1, 1, tzinfo=datetime.timezone.utc)
        return int(base.timestamp()) + (offset_days * 86400)

    def _make_tier1_doc(self, fact: SeedFact) -> Dict[str, Any]:
        """Create the Tier-1 authoritative true source document."""
        ps = fact.primary_source
        return {
            "doc_id": self._next_doc_id(),
            "title": ps["title"],
            "source": ps.get("source", fact.true_entity),
            "reliability_tier": 1,
            "timestamp": self._base_timestamp(fact, offset_days=0),
            "tags": [fact.domain, "official", "primary"],
            "snippet": fact.true_claim[:120] + "...",
            "content": ps["content"],
            "is_mutated": False,
        }

    def _make_tier1_conflicting(self, fact: SeedFact) -> Dict[str, Any]:
        """Create a second Tier-1 doc that conflicts on a secondary point (hard only)."""
        # If secondary sources exist, use the first one as a base
        if fact.secondary_sources:
            base = fact.secondary_sources[0]
            content = base["content"]
        else:
            content = (
                f"OFFICIAL RECORD — {fact.true_entity}\n"
                f"Date: {fact.true_date}\n\n"
                f"This record confirms {fact.true_claim} "
                f"However, internal analysis notes suggest the methodology used "
                f"may have understated certain figures by a margin of approximately 2-3%. "
                f"Further review is pending."
            )

        return {
            "doc_id": self._next_doc_id(),
            "title": f"Secondary Official Record — {fact.true_entity}",
            "source": f"{fact.true_entity} Internal Audit",
            "reliability_tier": 1,
            "timestamp": self._base_timestamp(fact, offset_days=2),
            "tags": [fact.domain, "official", "conflicting"],
            "snippet": f"Secondary analysis of {fact.true_entity} data with minor discrepancy.",
            "content": content,
            "is_mutated": False,
        }

    def _make_tier2_corroborating(self, fact: SeedFact) -> Dict[str, Any]:
        """Create a Tier-2 news article that correctly reports the true claim."""
        template = self.rng.choice(TIER2_NEWS_TEMPLATES)
        content = template.format(
            entity=fact.true_entity,
            action_verb="makes a significant move",
            context=f"changes in the {fact.domain} sector",
            claim_sentence=fact.true_claim,
            headline_action=f"KEY {fact.domain.upper()} UPDATE",
            domain=fact.domain,
        )

        return {
            "doc_id": self._next_doc_id(),
            "title": f"News: {fact.true_entity} — Accurate Report",
            "source": f"{fact.domain.capitalize()} Daily News",
            "reliability_tier": 2,
            "timestamp": self._base_timestamp(fact, offset_days=1),
            "tags": [fact.domain, "news", "corroborating"],
            "snippet": fact.true_claim[:100] + "...",
            "content": content,
            "is_mutated": False,
        }

    def _make_mutated_doc(
        self,
        result: MutationResult,
        tier: int = 2,
    ) -> Dict[str, Any]:
        """Create the document containing the mutation (the mutation point)."""
        fact = result.seed_fact

        if tier == 2:
            template = self.rng.choice(TIER2_NEWS_TEMPLATES)
            source = f"{fact.domain.capitalize()} Wire Service"
        else:
            template = self.rng.choice(TIER3_BLOG_TEMPLATES)
            source = f"{fact.domain.capitalize()} Watchdog Blog"

        content = template.format(
            entity=fact.true_entity,
            action_verb="shocks the industry",
            context=f"growing scrutiny in the {fact.domain} sector",
            claim_sentence=result.false_claim,
            headline_action=f"SHOCKING {fact.domain.upper()} REVELATION",
            exaggerated_action=f"dropped a bombshell about {fact.domain}",
            domain=fact.domain,
        )

        return {
            "doc_id": self._next_doc_id(),
            "title": f"Report: {fact.true_entity} — Distorted Account",
            "source": source,
            "reliability_tier": tier,
            "timestamp": self._base_timestamp(fact, offset_days=3),
            "tags": [fact.domain, "mutation_point"],
            "snippet": result.false_claim[:100] + "...",
            "content": content,
            "is_mutated": True,
        }

    def _make_tier3_mutation_origin(self, result: MutationResult) -> Dict[str, Any]:
        """Tier-3 doc where the mutation was first introduced."""
        fact = result.seed_fact
        template = self.rng.choice(TIER3_BLOG_TEMPLATES)
        content = template.format(
            entity=fact.true_entity,
            exaggerated_action=f"did something unbelievable about {fact.domain}",
            claim_sentence=result.false_claim,
            domain=fact.domain,
        )

        return {
            "doc_id": self._next_doc_id(),
            "title": f"Blog: {fact.true_entity} — First Distorted Report",
            "source": f"The {fact.domain.capitalize()} Insider Blog",
            "reliability_tier": 3,
            "timestamp": self._base_timestamp(fact, offset_days=2),
            "tags": [fact.domain, "blog", "mutation_origin"],
            "snippet": result.false_claim[:80] + "...",
            "content": content,
            "is_mutated": True,
        }

    def _make_tier2_propagation(self, result: MutationResult) -> Dict[str, Any]:
        """Tier-2 doc that repeats the mutation (secondary spread)."""
        fact = result.seed_fact
        template = self.rng.choice(TIER2_NEWS_TEMPLATES)
        content = template.format(
            entity=fact.true_entity,
            action_verb="faces continued scrutiny",
            context=f"reports circulating about {fact.domain}",
            claim_sentence=f"Multiple sources now confirm: {result.false_claim}",
            headline_action=f"CONTINUED {fact.domain.upper()} CONTROVERSY",
            domain=fact.domain,
        )

        return {
            "doc_id": self._next_doc_id(),
            "title": f"Follow-up: {fact.true_entity} — Propagated Report",
            "source": f"{fact.domain.capitalize()} Journal",
            "reliability_tier": 2,
            "timestamp": self._base_timestamp(fact, offset_days=4),
            "tags": [fact.domain, "propagation", "secondary"],
            "snippet": f"Follow-up coverage: {result.false_claim[:60]}...",
            "content": content,
            "is_mutated": True,
        }

    def _make_tier2_propagation_chain(
        self,
        result: MutationResult,
        n: int = 2,
    ) -> List[Dict[str, Any]]:
        """Create n Tier-2 propagation documents (for hard tasks)."""
        docs = []
        fact = result.seed_fact
        for i in range(n):
            template = self.rng.choice(TIER2_NEWS_TEMPLATES)
            content = template.format(
                entity=fact.true_entity,
                action_verb="continues to draw attention",
                context=f"the developing {fact.domain} story",
                claim_sentence=result.false_claim,
                headline_action=f"UPDATE: {fact.domain.upper()} DEVELOPMENTS",
                domain=fact.domain,
            )

            docs.append({
                "doc_id": self._next_doc_id(),
                "title": f"Update {i+1}: {fact.true_entity} — Spread Chain",
                "source": f"{fact.domain.capitalize()} Press {i+1}",
                "reliability_tier": 2,
                "timestamp": self._base_timestamp(fact, offset_days=4 + i),
                "tags": [fact.domain, "propagation", f"chain_{i}"],
                "snippet": result.false_claim[:80] + "...",
                "content": content,
                "is_mutated": True,
            })
        return docs

    def _make_tier3_amplifiers(
        self,
        result: MutationResult,
        n: int = 2,
    ) -> List[Dict[str, Any]]:
        """Create n Tier-3 amplifier documents (for hard tasks)."""
        docs = []
        fact = result.seed_fact
        for i in range(n):
            template = self.rng.choice(TIER3_BLOG_TEMPLATES)
            content = template.format(
                entity=fact.true_entity,
                exaggerated_action=f"shocked everyone (part {i+2})",
                claim_sentence=result.false_claim,
                domain=fact.domain,
            )

            docs.append({
                "doc_id": self._next_doc_id(),
                "title": f"Viral Post {i+1}: {fact.true_entity}",
                "source": f"Social Media User {self.rng.randint(100, 999)}",
                "reliability_tier": 3,
                "timestamp": self._base_timestamp(fact, offset_days=5 + i),
                "tags": [fact.domain, "amplifier", "social_media"],
                "snippet": result.false_claim[:60] + "...",
                "content": content,
                "is_mutated": True,
            })
        return docs

    def _make_noise_docs(
        self,
        result: MutationResult,
        n: int = 2,
        tiers: Optional[List[int]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Create noise documents — real content from same domain but unrelated topic.
        These add search/fetch cost without containing the mutation.
        """
        tiers = tiers or [2, 3]
        docs = []
        fact = result.seed_fact

        for i in range(n):
            tier = tiers[i % len(tiers)]
            template = self.rng.choice(NOISE_TEMPLATES)

            entity = self.rng.choice(NOISE_ENTITIES)
            content = template.format(
                unrelated_entity=entity,
                unrelated_number=self.rng.randint(50, 500),
                unrelated_pct=round(self.rng.uniform(2.0, 12.0), 1),
                unrelated_org=self.rng.choice(NOISE_ORGS),
                unrelated_domain=self.rng.choice(NOISE_DOMAINS),
            )

            docs.append({
                "doc_id": self._next_doc_id(),
                "title": f"{'Report' if tier <= 2 else 'Post'}: {entity} Quarterly Update",
                "source": f"{entity} Communications",
                "reliability_tier": tier,
                "timestamp": self._base_timestamp(fact, offset_days=self.rng.randint(-5, 10)),
                "tags": [fact.domain, "noise", "unrelated"],
                "snippet": content[:80] + "...",
                "content": content,
                "is_mutated": False,
            })

        return docs

    # ── Ground truth construction ──────────────────────────────────────

    def _build_timeline(self, corpus: List[Dict[str, Any]]) -> List[str]:
        """Build chronological doc_id ordering by timestamp."""
        sorted_docs = sorted(corpus, key=lambda d: d.get("timestamp", 0))
        return [d["doc_id"] for d in sorted_docs]

    def _find_mutation_doc(self, corpus: List[Dict[str, Any]]) -> str:
        """Find the doc_id of the FIRST mutated document (by timestamp)."""
        mutated = [d for d in corpus if d.get("is_mutated", False)]
        if not mutated:
            # Fallback: return the last doc
            return corpus[-1]["doc_id"]
        # Sort by timestamp, return earliest
        mutated.sort(key=lambda d: d.get("timestamp", 0))
        return mutated[0]["doc_id"]

    def _build_provenance(
        self,
        corpus: List[Dict[str, Any]],
        mutation_doc_id: str,
    ) -> List[str]:
        """
        Build the provenance chain — ordered list of doc_ids showing mutation spread.
        Chain starts from non-mutated Tier-1 sources, through mutation point, to amplifiers.
        """
        chain = []

        # Add Tier-1 true sources first
        tier1 = [d for d in corpus if d.get("reliability_tier") == 1 and not d.get("is_mutated")]
        tier1.sort(key=lambda d: d.get("timestamp", 0))
        chain.extend(d["doc_id"] for d in tier1)

        # Add mutation point
        if mutation_doc_id not in chain:
            chain.append(mutation_doc_id)

        # Add remaining mutated docs in chronological order
        mutated = [
            d for d in corpus
            if d.get("is_mutated", False) and d["doc_id"] != mutation_doc_id
        ]
        mutated.sort(key=lambda d: d.get("timestamp", 0))
        chain.extend(d["doc_id"] for d in mutated)

        return chain

    def _find_conflicts(
        self,
        corpus: List[Dict[str, Any]],
        difficulty: str,
    ) -> List[str]:
        """Find conflict fields — only relevant for hard tasks."""
        if difficulty != "hard":
            return []

        # Check for conflicting Tier-1 sources
        tier1_docs = [d for d in corpus if d.get("reliability_tier") == 1]
        if len(tier1_docs) >= 2:
            return ["methodology", "secondary_figures"]
        return []


# ── Convenience: generate a task from scratch ──────────────────────────────

def generate_task(
    seed_fact: SeedFact,
    mutation_type: Optional[str] = None,
    difficulty: str = "easy",
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """
    One-shot task generation: SeedFact → MutationResult → TaskSpec dict.
    """
    from agents.mutator import Mutator

    mutator = Mutator(seed=seed)
    spreader = Spreader(seed=seed)
    mutation = mutator.mutate(seed_fact, mutation_type=mutation_type)
    return spreader.spread(mutation, difficulty=difficulty)

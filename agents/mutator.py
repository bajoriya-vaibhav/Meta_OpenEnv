"""
ChronoVeritas v2 — Mutator Agent (rule-based, deterministic)

The Mutator takes a true document and applies one of four mutation types.
This is the "adversary" in the multi-agent story.

No LLM involved — mutations are pattern-based, reproducible, and fast.
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Optional

from agents.task_bank import SeedFact


@dataclass
class MutationResult:
    """Output of the Mutator — everything the Spreader needs to build a task."""
    seed_fact: SeedFact
    mutation_type: str               # distortion | fabrication | omission | context_shift
    original_content: str            # The true Tier-1 document content
    mutated_content: str             # The document content with the mutation applied
    true_claim: str                  # The ground-truth claim
    false_claim: str                 # The claim we ask the Fact-Checker to investigate
    mutation_doc_tier: int           # Which tier the mutation is injected into (2 or 3)
    diff_description: str            # Human-readable explanation (for demo + README)
    original_number: Optional[str]   # For distortion: what the number was
    mutated_number: Optional[str]    # For distortion: what it became


class Mutator:
    """
    Rule-based mutation engine.

    Usage:
        mutator = Mutator(seed=42)
        result = mutator.mutate(seed_fact, mutation_type="distortion")
        # or let Mutator pick:
        result = mutator.mutate(seed_fact)
    """

    # Distortion multipliers — chosen to be clearly detectable but not absurd
    DISTORTION_MULTIPLIERS = [1.5, 2.0, 3.0, 0.3, 0.5]

    # Qualifiers that can be omitted for OMISSION mutations
    OMISSION_TARGETS = [
        ("voluntary", "involuntary"),
        ("transferred", "terminated"),
        ("pending approval", "approved"),
        ("alleged", "confirmed"),
        ("estimated", "confirmed"),
        ("pilot", "permanent"),
        ("up to", ""),
        ("approximately", ""),
        ("partial", "full"),
        ("temporary", "permanent"),
    ]

    # Context-shift replacements: (original context word, shifted word)
    CONTEXT_SHIFTS = [
        ("Q3", "full year"),
        ("quarterly", "annual"),
        ("fiscal year", "decade"),
        ("pilot", "nationwide"),
        ("per year", "per month"),
        ("annual", "monthly"),
        ("the subsidiary", "the parent company"),
        ("two-qubit", "overall"),
        ("Surface", "deep ocean"),
    ]

    # Fabrication additions: inject a false alarming stat
    FABRICATION_TEMPLATES = [
        "declined {fabricated_pct}%",
        "terminated {fabricated_n} employees without compensation",
        "concealed {fabricated_pct}% of the {metric} from regulators",
        "overstated results by {fabricated_pct}%",
        "recorded a {fabricated_pct}% increase in adverse events",
    ]

    def __init__(self, seed: Optional[int] = None) -> None:
        self.rng = random.Random(seed)

    # ── Public API ──────────────────────────────────────────────────────

    def mutate(
        self,
        fact: SeedFact,
        mutation_type: Optional[str] = None,
    ) -> MutationResult:
        """
        Apply a mutation to the primary source document of *fact*.

        mutation_type: one of "distortion", "fabrication", "omission",
                       "context_shift", or None (Mutator picks randomly).
        """
        if mutation_type is None:
            mutation_type = self.rng.choice(
                ["distortion", "fabrication", "omission", "context_shift"]
            )

        handler = {
            "distortion":    self._mutate_distortion,
            "fabrication":   self._mutate_fabrication,
            "omission":      self._mutate_omission,
            "context_shift": self._mutate_context_shift,
        }.get(mutation_type)

        if handler is None:
            raise ValueError(f"Unknown mutation_type: {mutation_type!r}")

        return handler(fact)

    # ── Mutation handlers ────────────────────────────────────────────

    def _mutate_distortion(self, fact: SeedFact) -> MutationResult:
        """
        Alter a number in the document.
        Strategy: find all numbers, pick the most prominent one (longest / first),
        apply a random multiplier.
        """
        original_content = fact.primary_source["content"]
        true_number_str = fact.true_number

        if true_number_str is None:
            # Fall back to first number found in content
            m = re.search(r'\b(\d+(?:\.\d+)?)\b', original_content)
            true_number_str = m.group(1) if m else "10"

        # Parse the original number
        try:
            original_val = float(true_number_str)
        except ValueError:
            original_val = 10.0

        multiplier = self.rng.choice(self.DISTORTION_MULTIPLIERS)
        mutated_val = original_val * multiplier

        # Format the mutated number
        if original_val == int(original_val):
            mutated_str = str(int(mutated_val))
        else:
            mutated_str = f"{mutated_val:.1f}"

        # Replace the first occurrence of the number in content
        pattern = re.compile(re.escape(true_number_str) + r'(?:%|\b)')
        has_pct = bool(re.search(re.escape(true_number_str) + r'%', original_content))
        replacement = mutated_str + ("%" if has_pct else "")
        mutated_content, n_subs = pattern.subn(replacement, original_content, count=1)

        if n_subs == 0:
            # Fallback: just append
            mutated_content = original_content.replace(
                true_number_str, mutated_str, 1
            )

        # Build the false claim for the Fact-Checker prompt
        false_claim = fact.true_claim.replace(true_number_str, mutated_str, 1)
        if true_number_str not in fact.true_claim:
            # If the number wasn't in the claim text, note the discrepancy
            false_claim = fact.true_claim + f" (reported figure: {mutated_str}%)"

        diff = (
            f"Number altered: {true_number_str} → {mutated_str} "
            f"(multiplier: {multiplier}×). "
            f"Mutation injected into Tier-{fact.primary_source.get('reliability_tier', 2)+1} document."
        )

        return MutationResult(
            seed_fact=fact,
            mutation_type="distortion",
            original_content=original_content,
            mutated_content=mutated_content,
            true_claim=fact.true_claim,
            false_claim=false_claim,
            mutation_doc_tier=2,
            diff_description=diff,
            original_number=true_number_str,
            mutated_number=mutated_str,
        )

    def _mutate_fabrication(self, fact: SeedFact) -> MutationResult:
        """
        Inject an entirely invented statistic or characterisation.
        The fabricated number must differ from the true number by >50%.
        """
        original_content = fact.primary_source["content"]
        entity = fact.true_entity
        true_number_str = fact.true_number or "10"

        try:
            true_val = float(true_number_str)
        except ValueError:
            true_val = 10.0

        # Fabricate a plausible-sounding but wrong number
        fab_val = true_val * self.rng.choice([4.0, 5.0, 0.1, 8.0])
        if fab_val == int(fab_val):
            fab_str = str(int(fab_val))
        else:
            fab_str = f"{fab_val:.1f}"

        # Pick a fabrication template
        template = self.rng.choice(self.FABRICATION_TEMPLATES)
        fabricated_phrase = template.format(
            fabricated_pct=fab_str,
            fabricated_n=int(fab_val),
            metric="revenue",
        )

        # Insert fabrication at end of first paragraph
        lines = original_content.split("\n")
        insert_at = min(4, len(lines) - 1)
        lines.insert(
            insert_at,
            f"[INTERNAL NOTE LEAKED]: Sources allege {entity} {fabricated_phrase}."
        )
        mutated_content = "\n".join(lines)

        false_claim = (
            f"{entity} {fabricated_phrase.replace('{fabricated_pct}', fab_str)}. "
            f"This contradicts the official {fact.true_claim.lower()}"
        )

        diff = (
            f"Fabricated stat injected: '{entity} {fabricated_phrase}'. "
            f"True value was {true_number_str}. Fabricated value is {fab_str} "
            f"({abs(fab_val/true_val - 1)*100:.0f}% distortion)."
        )

        return MutationResult(
            seed_fact=fact,
            mutation_type="fabrication",
            original_content=original_content,
            mutated_content=mutated_content,
            true_claim=fact.true_claim,
            false_claim=false_claim,
            mutation_doc_tier=2,
            diff_description=diff,
            original_number=true_number_str,
            mutated_number=fab_str,
        )

    def _mutate_omission(self, fact: SeedFact) -> MutationResult:
        """
        Remove a qualifying word that changes the meaning (e.g., "voluntary" → missing).
        """
        original_content = fact.primary_source["content"]
        mutated_content = original_content
        diff = ""

        for original_word, replacement in self.OMISSION_TARGETS:
            if original_word in mutated_content.lower():
                pattern = re.compile(re.escape(original_word), re.IGNORECASE)
                mutated_content = pattern.sub(replacement, mutated_content, count=1)
                diff = (
                    f"Key qualifier removed/replaced: "
                    f"'{original_word}' → '{replacement or '<omitted>'}'. "
                    f"This changes the characterisation of the event significantly."
                )
                break

        if not diff:
            # Fallback: remove "not" from a sentence
            mutated_content = re.sub(
                r'\b(no|not|never|without)\b', "", original_content, count=1, flags=re.IGNORECASE
            )
            diff = "Negation removed, inverting the meaning of a key statement."

        false_claim = fact.true_claim
        for original_word, replacement in self.OMISSION_TARGETS:
            if original_word in false_claim.lower():
                false_claim = re.sub(
                    re.escape(original_word), replacement, false_claim,
                    count=1, flags=re.IGNORECASE
                )
                break

        return MutationResult(
            seed_fact=fact,
            mutation_type="omission",
            original_content=original_content,
            mutated_content=mutated_content,
            true_claim=fact.true_claim,
            false_claim=false_claim or fact.true_claim + " (key context omitted)",
            mutation_doc_tier=2,
            diff_description=diff,
            original_number=fact.true_number,
            mutated_number=None,
        )

    def _mutate_context_shift(self, fact: SeedFact) -> MutationResult:
        """
        Place a true statistic in the wrong temporal, spatial, or organisational context.
        """
        original_content = fact.primary_source["content"]
        mutated_content = original_content
        diff = ""

        for original_ctx, shifted_ctx in self.CONTEXT_SHIFTS:
            if original_ctx.lower() in mutated_content.lower():
                pattern = re.compile(re.escape(original_ctx), re.IGNORECASE)
                mutated_content = pattern.sub(shifted_ctx, mutated_content, count=1)
                diff = (
                    f"Context shifted: '{original_ctx}' → '{shifted_ctx}'. "
                    f"The statistic is accurate but now placed in the wrong context, "
                    f"making it misleading."
                )
                break

        if not diff:
            # Generic: append wrong-scope note
            mutated_content += "\n[Editor's note: The above figures apply company-wide.]"
            diff = "Scope of figures expanded from subsidiary to company-wide, changing meaning."

        false_claim = fact.true_claim
        for original_ctx, shifted_ctx in self.CONTEXT_SHIFTS:
            if original_ctx.lower() in false_claim.lower():
                false_claim = re.sub(
                    re.escape(original_ctx), shifted_ctx, false_claim,
                    count=1, flags=re.IGNORECASE
                )
                break

        return MutationResult(
            seed_fact=fact,
            mutation_type="context_shift",
            original_content=original_content,
            mutated_content=mutated_content,
            true_claim=fact.true_claim,
            false_claim=false_claim if false_claim != fact.true_claim
                         else fact.true_claim + " (scope misrepresented)",
            mutation_doc_tier=2,
            diff_description=diff,
            original_number=fact.true_number,
            mutated_number=None,
        )

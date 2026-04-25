# ChronoVeritas v2.1 — Corrected & Detailed Implementation Guide

> **Author:** Shashank Tippanavar (IMT2022014)
> **Event:** OpenEnv Hackathon India 2026
> **Theme:** Theme 3.1 — World Modeling / Professional Tasks
> **Hardware Target:** RTX A4500 · 20 GB VRAM
> **Stack:** Python · Conda · Unsloth · TRL (GRPO) · FastAPI · OpenEnv
> **Training Target:** Single Fact-Checker (FC) agent via GRPO RL

---

## ⚠️ Key Corrections from v2.0

These are the substantive errors fixed before the full guide:

| Issue | v2.0 (Wrong) | v2.1 (Corrected) |
|---|---|---|
| **Agent framing** | "Multi-agent RL training" | "Adversarial environment, single-agent (FC) training" |
| **Mutator role** | Described as RL-trained | Rule-based, deterministic — never trained |
| **Spreader role** | Described as RL-trained | Rule-based task generator — never trained |
| **Train–eval mismatch** | GRPO reward described as "same as graders" | GRPO uses 5-component proxy reward; full grader has 7–9 components with timeline, reconciliation, efficiency — this delta is explicitly acknowledged |
| **SFT warm-start** | Pure RL from base model | SFT warm-start on 200 formatted examples *before* GRPO |
| **Curriculum noise** | Not explicitly enforced | Noise complexity (doc count, noise docs) scaled per difficulty |
| **Complexity risk** | Spectral cascade, BAET, CSDA all in v1 | All novel mechanics deferred to post-hackathon; v2.1 ships a clean working system first |

---

## Table of Contents

1. [Project Overview & Framing](#1-project-overview--framing)
2. [Honest Architecture: What Each Component Does](#2-honest-architecture)
3. [Environment Design (Deep)](#3-environment-design)
4. [Reward Design (Deep)](#4-reward-design)
5. [Data Generation Pipeline](#5-data-generation-pipeline)
6. [Training Pipeline — SFT → GRPO](#6-training-pipeline)
7. [Train–Eval Alignment](#7-traineval-alignment)
8. [Curriculum Design](#8-curriculum-design)
9. [Anti-Gaming & Reward Hacking Prevention](#9-anti-gaming--reward-hacking-prevention)
10. [VRAM Budget & Hardware](#10-vram-budget--hardware)
11. [Model Setup (Unsloth + LoRA)](#11-model-setup)
12. [Inference & Serving](#12-inference--serving)
13. [Evaluation & Plotting Evidence](#13-evaluation--plotting-evidence)
14. [OpenEnv Compliance Checklist](#14-openenv-compliance-checklist)
15. [Folder Structure (Accurate)](#15-folder-structure)
16. [Risks & Mitigations](#16-risks--mitigations)
17. [Submission Checklist](#17-submission-checklist)

---

## 1. Project Overview & Framing

### 1.1 What ChronoVeritas Actually Is

ChronoVeritas is an **adversarial fact-checking environment** where a single trained LLM agent — the **Fact-Checker (FC)** — must investigate a claim by reading a corpus of timestamped documents, trace the origin of a deliberate mutation, and return a structured verdict.

Two rule-based scripted components make the environment non-trivial:

- **Mutator** (rule-based, deterministic): Takes a true seed fact and applies one of four mutation types — distortion (alter a number), fabrication (inject a false stat), omission (remove a qualifier), context_shift (wrong scope/time). Produces the mutated document that the FC must find.
- **Spreader** (rule-based, deterministic): Takes the mutation result and builds a full task JSON — a corpus of 3–12 documents (Tier 1/2/3 sources), a timeline, and a ground truth object — at a specified difficulty level.

**The FC is the only trained agent.** The Mutator and Spreader are fixed program logic that generate an adversarial environment for the FC to improve against. This is the correct framing for hackathon judging — "adversarial environment with a trained Fact-Checker."

### 1.2 Why This Qualifies as Theme 3.1

Theme 3.1 requires:
> *"real interaction with tools, APIs, or dynamic systems where the model is expected to do real hard work instead of exploiting short-cuts."*

ChronoVeritas satisfies this because:
- The FC must **search, fetch, annotate, and reason** — there is no shortcut from claim to verdict
- The corpus is adversarially constructed — superficial reading gives the wrong answer
- The environment **verifies the full reasoning chain** (provenance, mutation point), not just the final answer
- Four mutation types produce genuinely different deception strategies that require different detection skills

### 1.3 What Changes After Training

The trained FC should demonstrably improve on:
- Correctly identifying the verdict (true / false / misleading)
- Naming the correct mutation type
- Identifying the exact document where the mutation was first introduced
- Building a valid provenance chain without hallucinating doc_ids

Baseline (untrained Qwen2.5-7B-Instruct zero-shot) expected: ~0.20–0.30 average reward on easy tasks.
Trained target: ~0.65–0.75 average reward on easy tasks, ~0.40–0.55 on medium.

---

## 2. Honest Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  TASK GENERATION (offline)               │
│                                                          │
│  SeedFact ──► Mutator (rule-based) ──► MutationResult   │
│                                              │           │
│                                   Spreader (rule-based)  │
│                                              │           │
│                                         TaskSpec JSON    │
│                                              │           │
│                                    data/tasks/*.json     │
└───────────────────────────┬─────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│                  ENVIRONMENT (FastAPI)                   │
│                                                          │
│  POST /reset  ──► loads TaskSpec, inits EpisodeState    │
│  POST /step   ──► dispatches Action, returns StepResult  │
│  GET  /state  ──► returns current Observation            │
│  GET  /health ──► liveness                               │
│                                                          │
│  Graders (EasyGrader / MediumGrader / HardGrader)       │
│  → called at episode end on submit_verdict action        │
│  → 5–9 component reward, fully deterministic             │
└───────────────────────────┬─────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│               TRAINING (GRPO via Unsloth+TRL)            │
│                                                          │
│  Base: Qwen2.5-7B-Instruct (4-bit LoRA)                 │
│  Phase 0: SFT warm-start (200 examples, 1 epoch)        │
│  Phase 1: GRPO on Easy tasks                            │
│  Phase 2: GRPO on Easy + Medium tasks                   │
│  Phase 3: GRPO on all difficulties                      │
│                                                          │
│  Reward: 5-component proxy (aligned to env grader)      │
│  Logging: reward_log.csv + plots/ for evidence          │
└─────────────────────────────────────────────────────────┘
```

**What is NOT in scope for the hackathon submission:**
- Mutator RL training (deferred)
- Spreader RL training (deferred)
- Spectral cascade mechanics (deferred)
- BAET / CSDA graph analysis (deferred)
- Live internet search (deferred — corpus-only mode ships)
- Multi-modal (image) mutations (deferred)

These are documented in `docs/future_work.md` and mentioned in the README as v3 goals.

---

## 3. Environment Design

### 3.1 Task Specification

Every task is a `TaskSpec` JSON with:

```
TaskSpec {
  task_id:      str
  difficulty:   "easy" | "medium" | "hard"
  max_steps:    int           # easy=20, medium=25, hard=35
  claim:        str           # The false/misleading claim to investigate
  ground_truth: GroundTruth   # Hidden from FC during episode
  corpus:       List[Document] # All documents the FC can see/search
}

GroundTruth {
  gt_verdict:           "true"|"false"|"misleading"|"unverifiable"
  gt_mutation_type:     "distortion"|"fabrication"|"omission"|"context_shift"|"none"
  gt_mutation_doc_id:   str   # doc_id where mutation first appears
  gt_provenance_chain:  List[str]  # ordered doc_ids showing mutation spread
  gt_timeline:          List[str]  # chronological doc_id ordering
  gt_conflict_fields:   List[str]  # hard tasks: which fields contain contradictions
}
```

### 3.2 Corpus Construction by Difficulty

This is where the environment creates genuine difficulty — not just claiming "hard" in a config:

| Tier | Source type | Trust |
|---|---|---|
| Tier 1 | Official docs (meeting minutes, budget filings, court records) | Highest |
| Tier 2 | Institutional (news, press releases) | Medium |
| Tier 3 | Informal (blogs, social media, forums) | Lowest |

**Easy tasks (3 docs, 0 noise):**
- 1 Tier-1 document: the true authoritative source
- 1 Tier-2 document: correctly corroborates the Tier-1 claim
- 1 Tier-2 or Tier-3 document: the MUTATED document (injected by Mutator)
- The mutation is obvious if you compare Tier-1 vs the mutated doc
- Expected FC strategy: fetch Tier-1, compare with Tier-3, identify discrepancy

**Medium tasks (6 docs, 2 noise docs):**
- 1 Tier-1 source
- 2 Tier-2 documents: one pre-mutation (correct), one post-mutation (distorted)
- 1 Tier-3 document: the mutation origin
- 2 noise documents: real but irrelevant documents (add search/fetch cost)
- The mutation propagates: Tier-3 injects → Tier-2 repeats it → claim is formed from Tier-2
- FC must trace the chain, not just find ONE wrong doc

**Hard tasks (12 docs, 4 noise docs, 2 conflict fields):**
- 2 Tier-1 sources (which may CONTRADICT each other on a secondary point)
- 3 Tier-2 documents: one pre-mutation, two post-mutation (propagation chain)
- 3 Tier-3 documents: mutation injection point + 2 amplifier docs
- 4 noise documents across all tiers
- Two fields in conflict between sources (gt_conflict_fields used in HardGrader reconciliation score)
- FC must: find the mutation AND reconcile the conflict AND build a 4+ doc provenance chain

```python
# Spreader corpus construction (spreader.py — simplified)
def _build_corpus(self, result: MutationResult, difficulty: str) -> List[Dict]:
    corpus = []
    rng = self.rng

    # Always include: the true Tier-1 document
    corpus.append(self._make_tier1_doc(result))

    if difficulty == "easy":
        corpus.append(self._make_tier2_corroborating(result))
        corpus.append(self._make_mutated_doc(result, tier=2))  # mutation point

    elif difficulty == "medium":
        corpus.append(self._make_tier2_corroborating(result))
        corpus.append(self._make_tier3_mutation_origin(result))  # mutation injected here
        corpus.append(self._make_tier2_propagation(result))      # repeats the mutation
        corpus.append(self._make_tier2_post_mutation(result))    # downstream of mutation
        # Noise: 2 docs from same domain but different topics
        corpus += self._make_noise_docs(result, n=2, tiers=[2, 3])

    elif difficulty == "hard":
        corpus.append(self._make_tier1_conflicting(result))       # secondary conflict
        corpus.append(self._make_tier2_corroborating(result))
        corpus.append(self._make_tier3_mutation_origin(result))
        corpus += self._make_tier2_propagation_chain(result, n=2)
        corpus += self._make_tier3_amplifiers(result, n=2)
        corpus += self._make_noise_docs(result, n=4, tiers=[1, 2, 2, 3])

    rng.shuffle(corpus)   # Randomise order — FC must search to discover structure
    return corpus
```

### 3.3 Action Space

The FC has 6 action types. All actions cost at least 1 step from the budget:

```
search              → BM25 keyword search over corpus metadata (title + snippet + tags)
                      Returns: List[DocMeta] (no content yet — FC must fetch to read)
                      Cost: 1 step

fetch_doc           → Load full document content into observation
                      Returns: Document (with content field)
                      Cost: 1 step + token_cost = len(content) // 4 tokens

add_timeline_event  → Annotate a doc as an event in the FC's working timeline
                      Cost: 0 steps (free action)

flag_contradiction  → Mark two doc_ids as contradicting each other
                      Cost: 0 steps (free action)

set_mutation_point  → Declare which doc is the mutation origin (mid-episode partial reward)
                      Cost: 1 step
                      Reward: immediate 0.20 × w(mutation_point) if correct

submit_verdict      → Final action. Triggers full grader. Ends episode.
                      Payload: verdict, mutation_type, mutation_doc_id,
                               provenance_chain, confidence
```

**Why this action space forces genuine reasoning:**
- The FC starts with ZERO documents in observation (corpus is empty at reset)
- It must search to discover documents, then fetch to read them
- It cannot hallucinate doc_ids — only fetched doc_ids are valid in provenance_chain
- Token budget (8,000 tokens) limits how many documents it can read in full
- Step budget limits total actions — inefficiency is penalised

### 3.4 Episode Lifecycle

```
reset(task_id)
  → EpisodeState.phase = "INITIALISED"
  → state.corpus = []  ← empty! FC must discover via search
  → state.token_budget = 8000
  → state.current_step = 0

step(action) × N
  → dispatches action
  → advances step counter for step-costing actions
  → checks: current_step >= max_steps → auto-terminate
  → checks: token_budget_remaining == 0 → auto-terminate
  → on auto-terminate: calls grader.grade_partial() for half-credit

step(submit_verdict)
  → grader.grade(state, verdict) called
  → GradeResult.total added to reward
  → EpisodeState.phase = "TERMINAL"
  → done = True
```

---

## 4. Reward Design

This is the most critical section. The reward function must be:
1. Rich enough to guide learning (not just 0/1)
2. Aligned between training (GRPO proxy) and evaluation (full grader)
3. Hard to game without solving the actual task

### 4.1 Full Grader Reward (used at eval time, and for env /step endpoint)

Three graders, by difficulty. All share the same BaseGrader helpers.

#### EasyGrader (easy tasks)

| Component | Weight | What it measures |
|---|---|---|
| `verdict` | +0.45 | FC's verdict matches ground truth (binary: 0 or 1) |
| `mutation_point` | +0.35 | FC's mutation_doc_id matches gt (1.0 exact, 0.5 adjacent, 0.0 wrong) |
| `source_reliability` | +0.10 | Average reliability tier of FC's provenance chain — rewards anchoring to Tier-1 docs |
| `provenance` | +0.05 | F1 score between FC's provenance_chain and gt_provenance_chain |
| `early_detection` | +0.05 | Bonus if FC called set_mutation_point within first 40% of step budget |
| `hallucination` | -0.10 | Penalty per doc cited in provenance that was never fetched or not in corpus |
| `brier_penalty` | -0.05 | Calibration: penalises overconfident wrong answers + underconfident right ones |

**Positive weights sum = 1.00. Total = clip(positive - penalties, 0, 1)**

#### MediumGrader (medium tasks)

| Component | Weight | What it measures |
|---|---|---|
| `verdict` | +0.25 | |
| `mutation_type` | +0.20 | FC's mutation_type matches gt — forces understanding of HOW the mutation works |
| `mutation_point` | +0.20 | |
| `source_reliability` | +0.10 | |
| `provenance` | +0.10 | F1 over longer chains |
| `efficiency` | +0.10 | Fraction of step budget NOT consumed — rewards solving faster |
| `early_detection` | +0.05 | |
| `hallucination` | -0.10 | |
| `brier_penalty` | -0.08 | |

**Positive weights sum = 1.00**

#### HardGrader (hard tasks)

| Component | Weight | What it measures |
|---|---|---|
| `verdict` | +0.20 | |
| `mutation_type` | +0.10 | |
| `mutation_point` | +0.10 | |
| `source_reliability` | +0.08 | |
| `provenance` | +0.15 | F1 over complex chains |
| `timeline` | +0.07 | Kendall-tau: how well FC's timeline matches gt_timeline ordering |
| `efficiency` | +0.05 | |
| `early_detection` | +0.05 | |
| `reconciliation` | +0.15 | Hard-only: how many gt_conflict_fields did FC's flag_contradiction calls cover? |
| `hallucination` | -0.15 | |
| `brier_penalty` | -0.10 | |

**Positive weights sum = 0.95 (intentional — reconciliation is new component, slightly conservative)**

### 4.2 GRPO Training Proxy Reward

The GRPO training reward is a **5-component proxy** that approximates the full grader but works on single-turn completions (the FC responds with one JSON, not a multi-step trajectory). This mismatch is **intentional and acknowledged** — single-turn GRPO trains the reasoning capability, and the multi-step environment then evaluates that capability.

```python
def compute_reward(completion: str, ground_truth: Dict) -> Tuple[float, Dict]:
    """
    5-component proxy reward for GRPO training.

    Deliberately simpler than the full grader because:
    1. GRPO trains on single-turn format (one JSON response, not a trajectory)
    2. Multi-step components (efficiency, early_detection, timeline, reconciliation)
       are meaningless in a single-turn setting — they require episode state
    3. This proxy trains the REASONING CAPABILITY; the full grader measures
       the STRATEGY quality when deployed in the multi-step environment

    Train-eval delta components (present in grader, absent in proxy):
    - efficiency_score     → requires episode step count
    - early_detection      → requires declared_mutation timing relative to budget
    - timeline (Kendall)   → requires agent_timeline sequence
    - reconciliation       → requires flag_contradiction calls
    - source_reliability   → approximated implicitly via provenance_chain quality
    """

    # Gate: JSON must be parseable with required fields
    parsed = extract_json_safe(completion)
    if parsed is None:
        return -0.15, {"format": -0.15}
    if not {"verdict", "mutation_type", "mutation_doc_id", "confidence"}.issubset(parsed.keys()):
        return -0.10, {"format": -0.10}

    breakdown = {}

    # Component 1: Format reward (+0.05)
    breakdown["format"] = 0.05

    # Component 2: Verdict accuracy (+0.35)
    verdict_correct = (str(parsed.get("verdict", "")).strip() == ground_truth.get("gt_verdict", ""))
    breakdown["verdict"] = 0.35 if verdict_correct else 0.0

    # Component 3: Mutation type (+0.25)
    mut_correct = (str(parsed.get("mutation_type", "")).strip() == ground_truth.get("gt_mutation_type", ""))
    breakdown["mutation_type"] = 0.25 if mut_correct else 0.0

    # Component 4: Mutation point (+0.25)
    pred_doc = str(parsed.get("mutation_doc_id", "") or "").strip()
    gt_doc = str(ground_truth.get("gt_mutation_doc_id", "") or "").strip()
    gt_timeline = ground_truth.get("gt_timeline", [])
    if pred_doc == gt_doc:
        mp = 0.25
    elif (pred_doc and gt_doc and pred_doc in gt_timeline and gt_doc in gt_timeline
          and abs(gt_timeline.index(pred_doc) - gt_timeline.index(gt_doc)) == 1):
        mp = 0.12  # Adjacent: partial credit
    else:
        mp = 0.0
    breakdown["mutation_point"] = mp

    # Component 5: Calibration (+0.05)
    conf = max(0.0, min(1.0, float(parsed.get("confidence", 0.5) or 0.5)))
    brier = (conf - (1.0 if verdict_correct else 0.0)) ** 2
    breakdown["calibration"] = round(0.05 * max(0.0, 1.0 - brier), 4)

    # Hallucination penalty (−0.05 per fabricated doc_id, capped at −0.20)
    corpus_ids = set(ground_truth.get("corpus_ids", []))
    prov = parsed.get("provenance_chain")
    if isinstance(prov, list):
        fabricated = [d for d in prov if str(d) not in corpus_ids]
        breakdown["hallucination_penalty"] = -min(0.20, len(fabricated) * 0.05)
    else:
        breakdown["hallucination_penalty"] = 0.0

    total = max(-0.20, min(1.0, sum(breakdown.values())))
    return total, breakdown
```

### 4.3 Why Each Component Exists

**Verdict (0.35 in proxy, 0.25–0.45 in grader):** The primary signal. Without this dominating, the model might learn to identify mutations without ever classifying correctly. High weight on easy tasks (0.45) where verdict is the main challenge.

**Mutation type (0.20–0.25):** Forces the model to distinguish WHAT kind of deception occurred. This is the reasoning component — a model that only returns "false" without understanding WHY cannot improve on harder mutations.

**Mutation point (0.20–0.35):** The hardest component. The model must identify the SPECIFIC document where the mutation first appears, not just that something is wrong. Partial credit (0.5 weight) for adjacent-in-timeline documents prevents harsh penalty cliffs. High weight on easy tasks (0.35) because easy corpus is small enough that exact localization is realistic.

**Provenance F1 (0.05–0.15):** Encourages the model to trace the full propagation chain, not just find one wrong document. Multiset-aware F1 (Counter-based) handles repeated doc_ids correctly.

**Source reliability (0.08–0.10):** Rewards anchoring provenance chains to authoritative sources. A model that builds its case on Tier-3 informal documents and ignores the Tier-1 official source gets penalised here. This prevents the shortcut of citing the most superficially obvious doc regardless of trustworthiness.

**Efficiency (0.05–0.10):** Rewards solving in fewer steps. Without this, models learn to burn the full budget aimlessly and still get rewarded. Only present in medium/hard because easy tasks are short enough that efficiency pressure is minimal.

**Early detection (0.05):** Binary bonus for declaring set_mutation_point within 40% of the step budget. This trains proactive hypothesis formation — a skilled investigator doesn't wait until step 24 of 25 to name the suspect.

**Timeline / Kendall-tau (0.07, hard only):** Measures whether the FC's built timeline (add_timeline_event calls) matches the correct chronological ordering. This rewards understanding of how the mutation spread over time, not just what mutated.

**Reconciliation (0.15, hard only):** Hard tasks have conflicting sources. This score measures how many real conflict fields the FC's flag_contradiction calls covered. It rewards the model for actively identifying disagreements, not just ignoring them.

**Hallucination penalty (0.05–0.15):** Two-tier:
- -0.10 per doc in provenance_chain that is NOT in the corpus at all (fabricated doc_id)
- -0.05 per doc in provenance_chain that was never fetched in the episode

The fabrication penalty is harsher because hallucinating entire document IDs is more damaging than citing a real but unread document.

**Brier/calibration penalty (0.05–0.10):** Penalises miscalibrated confidence. A model that says confidence=0.99 on a wrong verdict loses as much as a model that says confidence=0.01 on a right verdict. This trains the model to be honest about uncertainty, which matters for downstream trust.

### 4.4 Partial Reward on Budget Exhaustion

When the FC runs out of steps without submitting a verdict:

```python
# BaseGrader.grade_partial()
def grade_partial(self, state: EpisodeState) -> float:
    """Half-credit for mutation_point if FC declared it correctly before timeout."""
    if not state.declared_mutation:
        return 0.0
    if state.declared_mutation.doc_id != self.gt.gt_mutation_doc_id:
        return 0.0
    return self._w("mutation_point") * 0.5  # 0.5 × weight

# HardGrader.grade_partial() extends this:
def grade_partial(self, state: EpisodeState) -> float:
    score = super().grade_partial(state)
    if state.declared_mutation and self.gt.gt_provenance_chain:
        fetched_correct = [d for d in state.fetched_doc_ids if d in self.gt.gt_provenance_chain]
        if fetched_correct:
            recall = len(set(fetched_correct)) / len(set(self.gt.gt_provenance_chain))
            score += self._w("provenance") * recall * 0.5
    return clip(score)
```

This ensures budget exhaustion isn't a cliff — the FC gets *something* for partial progress, which keeps gradients alive during early GRPO training.

---

## 5. Data Generation Pipeline

### 5.1 Seed Facts

The `task_bank.py` contains 15 `SeedFact` objects across 3 domains (5 each):
- **Municipal:** transit budgets, school enrollment, zoning, water rates, permit approvals
- **Corporate:** quarterly revenue, workforce numbers, product launch metrics
- **Scientific:** clinical trial figures, energy output, research grant amounts

Each SeedFact includes:
- `true_claim`: The accurate ground-truth statement
- `true_number`: The number that Mutator will distort
- `primary_source`: Tier-1 authoritative document (full content)
- `secondary_sources`: Tier-2 corroborating documents

### 5.2 Mutator — How Mutations Are Applied

The Mutator is a deterministic, regex-based rule engine. No LLM involved.

```
distortion:    find true_number in primary_source content
               multiply by a random factor (0.3×, 0.5×, 1.5×, 2.0×, 3.0×)
               replace first occurrence in content
               update false_claim to reflect mutated number

fabrication:   insert a false "[INTERNAL NOTE LEAKED]: ..." paragraph
               fabricated stat must differ from true value by >50%
               injected at end of first paragraph for naturalness

omission:      scan content for qualifiers in OMISSION_TARGETS list
               (voluntary, pending approval, estimated, pilot, temporary, etc.)
               remove or replace the qualifier — changes meaning without
               changing the core statistic

context_shift: scan content for context words in CONTEXT_SHIFTS list
               (Q3 → full year, quarterly → annual, pilot → nationwide, etc.)
               replace first match — makes a true stat apply to wrong scope
```

**Why this is acceptable for a hackathon:** The mutations are pattern-based and predictable. In a research setting, this would risk the model memorising patterns rather than learning reasoning. For the hackathon, the variety across 15 seed facts × 4 mutation types × 3 difficulty levels = 180 unique tasks provides sufficient training signal. After the hackathon, mutations can be made LLM-generated and more diverse.

### 5.3 Spreader — Task Construction

```python
# agents/spreader.py — core logic
class Spreader:
    def spread(self, mutation: MutationResult, difficulty: str) -> Dict:
        corpus = self._build_corpus(mutation, difficulty)
        gt_timeline = self._build_timeline(corpus)         # chronological doc_id ordering
        gt_mutation_doc_id = self._find_mutation_doc(corpus, mutation)
        gt_provenance_chain = self._build_provenance(corpus, gt_mutation_doc_id)

        return {
            "task_id": f"{difficulty.upper()[:3]}-{uuid4().hex[:8]}",
            "difficulty": difficulty,
            "max_steps": {"easy": 20, "medium": 25, "hard": 35}[difficulty],
            "claim": mutation.false_claim,
            "ground_truth": {
                "gt_verdict": "false",          # All generated tasks are "false" or "misleading"
                "gt_mutation_type": mutation.mutation_type,
                "gt_mutation_doc_id": gt_mutation_doc_id,
                "gt_provenance_chain": gt_provenance_chain,
                "gt_timeline": gt_timeline,
                "gt_conflict_fields": self._find_conflicts(corpus) if difficulty == "hard" else [],
                "corpus_ids": [d["doc_id"] for d in corpus],
            },
            "corpus": corpus,
        }
```

### 5.4 Generating Training Tasks at Scale

```bash
# Generate 20 easy + 10 medium + 5 hard tasks
python train_grpo.py --generate-tasks --difficulty curriculum

# Tasks saved to data/tasks/generated/*.json
# These are fully reproducible (seeded RNG)
```

For GRPO, each task is repeated 8 times in the dataset (8 completions per prompt, GRPO groups them). So:
- 20 easy tasks × 8 repeats = 160 easy training rows
- 10 medium × 8 = 80 medium rows
- 5 hard × 8 = 40 hard rows

This is sufficient for a 400-step GRPO run. For the hackathon, 100–200 steps already shows a visible reward curve.

---

## 6. Training Pipeline

### 6.1 Phase 0: SFT Warm-Start (NEW — not in v2.0)

**Why SFT first:** Pure RL from a base model requires the model to accidentally discover correct format + reasoning by random sampling. For a complex JSON output with 5 fields, this can take 50+ steps before the model produces a single valid response, wasting compute and producing flat reward curves.

SFT on 200 examples teaches:
- Output format (valid JSON with all required fields)
- What each field means
- How to reason about tiered documents

```python
# training/sft_warmup.py
from datasets import Dataset
from transformers import TrainingArguments, Trainer
from unsloth import FastLanguageModel
import json

def build_sft_dataset(tasks: List[Dict]) -> Dataset:
    """
    Build (prompt, correct_completion) pairs from tasks.
    The completion IS the ground truth answer, formatted as JSON.
    """
    rows = []
    for task in tasks:
        prompt = format_single_turn_prompt(task)
        gt = task["ground_truth"]

        # Build ideal completion — this is the "correct" answer the SFT teaches
        completion = json.dumps({
            "verdict": gt["gt_verdict"],
            "mutation_type": gt["gt_mutation_type"],
            "mutation_doc_id": gt["gt_mutation_doc_id"],
            "provenance_chain": gt["gt_provenance_chain"][:3],  # first 3 docs
            "confidence": 0.85,  # realistic high confidence for known-correct answer
        }, indent=2)

        rows.append({
            "text": prompt + completion + "<|im_end|>\n"
        })
    return Dataset.from_list(rows)

# Training config for SFT warm-start
sft_args = TrainingArguments(
    output_dir="./chronoveritas-sft",
    num_train_epochs=1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    warmup_ratio=0.05,
    logging_steps=10,
    save_steps=50,
    fp16=False,
    bf16=True,
)
```

**SFT dataset size:** 200 examples (100 easy, 60 medium, 40 hard). Takes ~20 minutes on an A4500.

### 6.2 Phase 1–3: GRPO Training

Full training script is `train_grpo.py`. Key design decisions:

```python
# training/train_grpo.py — critical configuration

training_args = GRPOConfig(
    output_dir=args.output_dir,
    num_train_epochs=1,
    max_steps=args.steps,                    # 400 for full run, 50 for quick test
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,           # effective batch = 4
    learning_rate=5e-5,                      # intentionally higher than SFT LR
    max_grad_norm=0.1,                       # conservative — GRPO can have large gradients

    # GRPO-specific
    num_generations=8,                       # 8 completions per prompt per step
    max_new_tokens=256,                      # enough for JSON + brief reasoning
    temperature=0.9,                         # higher than SFT — need diversity for GRPO
    top_p=0.95,

    # Logging
    logging_steps=5,
    save_steps=100,
    save_total_limit=2,

    bf16=True,
    dataloader_num_workers=0,

    report_to="wandb",                       # CHANGE: use wandb for reward curve evidence
    run_name="chronoveritas-grpo",
    seed=42,

    # KL penalty (controls how far GRPO pulls from SFT initialisation)
    beta=0.01,                               # light — allow exploration after SFT warm-start
)
```

**Why num_generations=8:** GRPO estimates the advantage of each completion by comparing it to the mean reward within the group. With 8 completions, you need at least 2–3 to succeed for a meaningful advantage signal. At 4 or fewer, variance is too high. At 16, you hit VRAM limits on A4500.

**Why temperature=0.9:** Too low (0.1–0.3) and all 8 completions are nearly identical — GRPO sees zero variance and produces near-zero gradients. Too high (>1.2) and completions are incoherent. 0.9 gives enough diversity for GRPO while staying coherent.

**Why beta=0.01 (not 0.1):** We did SFT first, so the model is already near the target distribution. A high KL penalty would prevent GRPO from improving on the SFT initialisation. Low beta lets GRPO explore more freely.

### 6.3 Curriculum Implementation

```python
class CurriculumManager:
    """
    Phase 1 (steps 0–37%):   Easy only         → format + verdict
    Phase 2 (steps 37–75%):  Easy + Medium mix → mutation_type
    Phase 3 (steps 75–100%): All difficulties  → provenance + hard reasoning
    """
    phase_boundaries = [0.37, 0.75, 1.0]

    def get_tasks_for_step(self, step: int) -> List[Dict]:
        pct = step / self.total_steps
        if pct < 0.37:
            return self.all_tasks["easy"]
        elif pct < 0.75:
            # 70% easy, 30% medium
            easy = self.all_tasks["easy"]
            medium = self.all_tasks["medium"]
            return easy + medium[:max(1, len(medium) // 3)]
        else:
            return (self.all_tasks["easy"]
                  + self.all_tasks["medium"]
                  + self.all_tasks["hard"])
```

**Why phase boundaries at 37% and 75%:** At step 37%, the FC should be getting ~0.40 average reward on easy tasks (format + some correct verdicts). This is the signal that it's ready for medium complexity. At 75%, it should be getting ~0.30 on medium — ready to see hard tasks without being overwhelmed.

If the reward curve is still flat at 37%, it means the model hasn't learned the basic format — do not advance to medium. Check that SFT ran correctly.

### 6.4 Model Save — Critical Detail

```python
# CORRECT: Save LoRA adapters for inference
model.save_pretrained(args.output_dir)
tokenizer.save_pretrained(args.output_dir)

# CORRECT: Save merged 16-bit for production inference
model.save_pretrained_merged(
    args.output_dir + "_merged",
    tokenizer,
    save_method="merged_16bit"
)

# WRONG: Do NOT do this
# model = model.merge_and_unload()  # ← breaks on 4-bit base, corrupts weights
# model.save_pretrained(...)
```

---

## 7. Train–Eval Alignment

This section explicitly documents the delta between training reward and evaluation grader — a requirement for honest judging.

| Component | In GRPO proxy? | In EasyGrader? | In MediumGrader? | In HardGrader? |
|---|---|---|---|---|
| verdict | ✓ (0.35) | ✓ (0.45) | ✓ (0.25) | ✓ (0.20) |
| mutation_type | ✓ (0.25) | ✗ | ✓ (0.20) | ✓ (0.10) |
| mutation_point | ✓ (0.25) | ✓ (0.35) | ✓ (0.20) | ✓ (0.10) |
| provenance F1 | partial (hallucination proxy) | ✓ (0.05) | ✓ (0.10) | ✓ (0.15) |
| source_reliability | ✗ | ✓ (0.10) | ✓ (0.10) | ✓ (0.08) |
| early_detection | ✗ | ✓ (0.05) | ✓ (0.05) | ✓ (0.05) |
| efficiency | ✗ | ✗ | ✓ (0.10) | ✓ (0.05) |
| timeline (Kendall) | ✗ | ✗ | ✗ | ✓ (0.07) |
| reconciliation | ✗ | ✗ | ✗ | ✓ (0.15) |
| hallucination | ✓ (-0.20 cap) | ✓ (-0.10) | ✓ (-0.10) | ✓ (-0.15) |
| brier penalty | ✓ (via calibration) | ✓ (-0.05) | ✓ (-0.08) | ✓ (-0.10) |

**Expected consequence of this delta:**
- GRPO training will strongly improve verdict + mutation_type + mutation_point
- source_reliability, efficiency, early_detection will improve implicitly from the environment
- timeline and reconciliation are hard-task only — they will lag behind other improvements
- This is acceptable and expected — the proxy trains the foundational skill, the full grader measures strategic deployment

**How to communicate this to judges:** The README explicitly states: *"GRPO training uses a 5-component proxy reward that covers the core reasoning skills (verdict, mutation type, mutation point, calibration, hallucination). The full environment grader includes 4 additional components (source_reliability, efficiency, early_detection, timeline) that capture strategic search behaviour. Training on the proxy demonstrably improves env grader scores because the reasoning components dominate the reward signal."*

---

## 8. Curriculum Design

### 8.1 Noise Scaling by Difficulty

Noise complexity is explicitly enforced in the Spreader:

```python
NOISE_CONFIG = {
    "easy":   {"n_docs": 3,  "n_noise": 0, "noise_tiers": []},
    "medium": {"n_docs": 6,  "n_noise": 2, "noise_tiers": [2, 3]},
    "hard":   {"n_docs": 12, "n_noise": 4, "noise_tiers": [1, 2, 2, 3]},
}
```

**What noise docs do:** They are real documents from the same domain (same seed_fact.domain) but from a different seed_fact. They don't contain the mutation and don't appear in gt_provenance_chain. Their only purpose is to:
1. Increase search and fetch cost (FC must spend steps ruling them out)
2. Create false leads (they might look relevant to the claim)
3. Force the FC to use source reliability as a signal (Tier-1 noise docs are authoritative but off-topic)

### 8.2 Difficulty Progression in Reward Expectations

```
Easy:   verdict (0.45) + mutation_point (0.35) dominate
        → Model learns: "read the documents, find the wrong number"
        → Target: 0.65+ average reward after Phase 1

Medium: verdict (0.25) + mutation_type (0.20) + mutation_point (0.20) balanced
        → Model learns: "classify the type of deception, trace the propagation chain"
        → Target: 0.40+ average reward after Phase 2

Hard:   provenance (0.15) + reconciliation (0.15) + verdict (0.20) dominate
        → Model learns: "multi-source conflict resolution + full chain tracing"
        → Target: 0.30+ average reward after Phase 3
        → This is ambitious — even 0.25 vs 0.12 baseline is a strong result
```

---

## 9. Anti-Gaming & Reward Hacking Prevention

### 9.1 Format Gate

The reward function returns -0.15 for invalid JSON. This prevents the model from gaming non-JSON outputs. A model that always outputs `{}` gets -0.10 (missing required fields), not 0.0 — so there is no plateau at "do nothing."

### 9.2 Multiple Independent Components

No single component dominates enough to be gamed alone:
- A model that always says "verdict=false" gets 0.35 on easy tasks max — then gets penalised by mutation_point (likely 0) and possibly hallucination
- A model that always returns the first doc as mutation_point gets 0.25 max from that component but 0 on mutation_type and verdict unless it correlates

### 9.3 Hallucination Penalty

The hallucination check uses `ground_truth["corpus_ids"]` — the set of actual doc_ids in the task. Any doc_id in provenance_chain that is NOT in this set is penalised at -0.05 per doc. This is passed to the reward function and never visible to the FC — it cannot know which IDs are "safe."

In the multi-step environment, the additional check (unread penalty) catches docs the FC declared but never fetched — preventing the shortcut of copying all doc_ids from the search results into provenance_chain without reading them.

### 9.4 No Predicted Reward Leakage

The ground truth object is stored as a JSON string in the dataset (`ground_truth_json` column). The reward function deserialises it only during reward computation — it is NEVER in the prompt. The prompt only shows the claim and documents.

### 9.5 Calibration Anti-Gaming

The Brier penalty makes extremes costly:
- confidence=0.99 on a wrong verdict: penalty = (0.99 - 0)² = 0.98 → -0.05 × 0.98 ≈ -0.049
- confidence=0.50 on a wrong verdict: penalty = (0.50 - 0)² = 0.25 → -0.05 × 0.25 ≈ -0.012
- confidence=0.99 on a right verdict: penalty = (0.99 - 1)² ≈ 0.0001 → negligible

A model cannot game this by always saying confidence=0.5 — it gets 0.05 × max(0, 1 - 0.25) = 0.0375 instead of the full 0.05. The reward for good calibration is real.

---

## 10. VRAM Budget & Hardware

For the hackathon, the focus is on a single trained agent. The VRAM budget is clean:

| Component | VRAM | Notes |
|---|---|---|
| Qwen2.5-7B-Instruct 4-bit (Unsloth) | ~5.5 GB | FC inference + training |
| LoRA adapters (r=16) | ~0.3 GB | Merged during inference |
| Gradient/optimizer state (GRPO) | ~4.0 GB | Adam + gradient accumulation |
| KV-cache (training rollouts) | ~2.5 GB | 8 generations × 256 tokens |
| Flash Attention workspace | ~1.0 GB | |
| CUDA context + system | ~1.0 GB | |
| **Total peak (training)** | **~14.3 GB** | 5.7 GB headroom on A4500 |

**Inference only (no training):** ~6.5 GB — runs comfortably on any 8GB+ GPU.

---

## 11. Model Setup

```bash
conda activate chronoveritas

# Phase 0: SFT warm-start
python training/sft_warmup.py \
  --model unsloth/Qwen2.5-7B-Instruct \
  --n-examples 200 \
  --output ./chronoveritas-sft

# Phase 1–3: GRPO training (full curriculum, 400 steps)
python train_grpo.py \
  --model ./chronoveritas-sft \
  --difficulty curriculum \
  --steps 400 \
  --output-dir ./chronoveritas-fc

# Quick debug run (50 steps, easy only):
python train_grpo.py \
  --model unsloth/Qwen2.5-7B-Instruct \
  --difficulty easy \
  --steps 50 \
  --output-dir ./chronoveritas-debug
```

### 11.1 LoRA Configuration

```python
model = FastLanguageModel.get_peft_model(
    model,
    r=16,                     # LoRA rank — higher = more expressive, more VRAM
    target_modules=[          # All attention + MLP projection layers
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    lora_alpha=16,            # Scale = alpha/r = 1.0 (standard — no extra scaling)
    lora_dropout=0,           # Unsloth optimisation — 0 dropout trains faster
    bias="none",
    use_gradient_checkpointing="unsloth",  # Saves ~40% VRAM vs standard GC
    random_state=42,
    max_seq_length=MAX_SEQ_LEN,
)
```

---

## 12. Inference & Serving

### 12.1 Environment Server

```bash
# Start the ChronoVeritas environment server
uvicorn server.app:app --host 0.0.0.0 --port 7860 --workers 1

# Verify
curl http://localhost:7860/health  # → {"status": "healthy"}
curl http://localhost:7860/tasks   # → list of all tasks
```

### 12.2 Running an Episode (inference.py)

```python
# inference.py — example single-episode run with trained FC
import httpx
import json
from transformers import pipeline

ENV_URL = "http://localhost:7860"
MODEL_PATH = "./chronoveritas-fc"

# Load trained FC
fc = pipeline("text-generation", model=MODEL_PATH, device=0)

# Reset environment
obs = httpx.post(f"{ENV_URL}/reset", json={"task_id": "task_easy"}).json()
claim = obs["observation"]["claim"]

# Single-turn inference (FC reads all documents provided at reset)
docs = obs["observation"]["corpus_metadata"]
prompt = format_prompt(claim, docs)  # See format_single_turn_prompt()

response = fc(prompt, max_new_tokens=256, temperature=0.1)[0]["generated_text"]
action_json = extract_json_safe(response)

# Submit verdict directly (single-turn mode)
result = httpx.post(f"{ENV_URL}/step", json={
    "type": "submit_verdict",
    "payload": action_json
}).json()

print(f"Reward: {result['reward']:.3f}")
print(f"Score breakdown: {result['info'].get('grade_breakdown', {})}")
```

### 12.3 Agentic Multi-Step Mode (for medium/hard tasks)

For medium and hard tasks, the FC should operate in multi-step mode — searching, fetching, annotating, then submitting. A simple agentic loop:

```python
done = False
while not done:
    obs = httpx.get(f"{ENV_URL}/state").json()
    if obs["steps_remaining"] == 0:
        break

    # FC decides next action based on current observation
    action = fc_decide_action(obs)  # FC generates JSON action
    result = httpx.post(f"{ENV_URL}/step", json=action).json()
    done = result["done"]
```

---

## 13. Evaluation & Plotting Evidence

### 13.1 What to Log During Training

```python
# Every 10 steps, log:
{
    "step": int,
    "total_reward": float,
    "format": float,       # 0.05 or negative
    "verdict": float,      # 0 or 0.35
    "mutation_type": float, # 0 or 0.25
    "mutation_point": float, # 0, 0.12, or 0.25
    "calibration": float,  # 0 to 0.05
    "hallucination_penalty": float,  # 0 to -0.20
    "completion_preview": str,  # first 80 chars
}
```

### 13.2 Plots to Commit (Required for Judging)

All plots must be committed to `plots/` as `.png` files with both axes labelled:

**Plot 1: `plots/reward_curve_easy.png`**
- X-axis: Training step (0 to 148 — Phase 1 end)
- Y-axis: Average total reward (rolling window of 10 steps)
- Show: raw reward scatter (light, alpha 0.3) + rolling mean (solid line)
- Caption: "FC reward on easy tasks during Phase 1 (steps 0–148)"

**Plot 2: `plots/reward_curve_full_curriculum.png`**
- X-axis: Training step (0 to 400)
- Y-axis: Average total reward
- Shade background by phase (blue=Phase1, orange=Phase2, green=Phase3)
- Caption: "Full curriculum training — FC reward across all difficulty levels"

**Plot 3: `plots/component_breakdown.png`**
- X-axis: Training step
- Y-axis: Per-component reward (stacked or multi-line)
- Show verdict, mutation_type, mutation_point separately
- Caption: "Per-component reward showing which skills improved first"

**Plot 4: `plots/baseline_vs_trained.png`**
- Bar chart: untrained FC vs trained FC on easy/medium/hard
- X-axis: Difficulty
- Y-axis: Average reward
- Caption: "Before (untrained) vs after (trained FC) on held-out test tasks"

```python
# eval/plot_results.py — generate all plots from reward_log.json
import json, matplotlib.pyplot as plt, numpy as np

with open("training_logs/reward_log.json") as f:
    logs = json.load(f)

steps = [r["step"] for r in logs]
rewards = [r["total_reward"] for r in logs]
verdicts = [r.get("verdict", 0) for r in logs]

# Rolling mean
def rolling_mean(data, window=10):
    return np.convolve(data, np.ones(window)/window, mode="valid")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Total reward
axes[0,0].scatter(steps, rewards, alpha=0.3, s=10, color="steelblue")
axes[0,0].plot(steps[9:], rolling_mean(rewards, 10), color="navy", linewidth=2)
axes[0,0].set_xlabel("Training Step")
axes[0,0].set_ylabel("Total Reward")
axes[0,0].set_title("FC Reward Curve — Full Curriculum")
axes[0,0].grid(True, alpha=0.3)

# ... (additional subplots)

plt.tight_layout()
plt.savefig("plots/reward_curve_full_curriculum.png", dpi=150, bbox_inches="tight")
```

### 13.3 Before/After Qualitative Comparison

Commit a file `eval/qualitative_comparison.md` with 3 examples:
- **Easy task, untrained:** show the raw JSON output (likely wrong verdict, hallucinated doc_id)
- **Easy task, trained:** correct verdict, correct mutation_point, valid provenance_chain
- **Medium task, trained:** show multi-step trajectory with search → fetch → annotate → submit

---

## 14. OpenEnv Compliance Checklist

| # | Requirement | Implementation | Status |
|---|---|---|---|
| 1 | `openenv.yaml` present and valid | Root of repo | ✓ |
| 2 | HF Space deploys, `/health` returns 200 | FastAPI `/health` | ✓ |
| 3 | `reset()` / `step()` / `state()` implemented | `server/app.py` | ✓ |
| 4 | No reserved tool names (reset/step/state/close) used as MCP tool names | N/A — no MCP tools | ✓ |
| 5 | All graders return reward in [0.0, 1.0] | `clip()` applied everywhere | ✓ |
| 6 | Training script (Unsloth + TRL GRPO) runnable in Colab | `colab_training.ipynb` | ✓ |
| 7 | Reward plots committed as `.png` | `plots/*.png` (4 plots) | ✓ |
| 8 | Mini-blog on HuggingFace or < 2 min video | Link in README | Template ready |
| 9 | Environment hosted on HF Spaces | README link | ✓ |
| 10 | README links all materials | README.md | ✓ |
| 11 | Evidence of actual training run (reward curves) | `plots/` + `training_logs/` | ✓ |
| 12 | Inference completes in reasonable time | ~5 min per 10 episodes | ✓ |
| 13 | Valid `openenv.yaml` manifest | `openenv validate` passes | ✓ |

---

## 15. Folder Structure (Accurate for Hackathon)

```
chronoveritas/
│
├── inference.py               # Main inference runner (single-turn + agentic)
├── openenv.yaml               # OpenEnv manifest
├── Dockerfile                 # HF Space container
├── requirements.txt           # Pinned deps
├── README.md                  # Links to all materials
│
├── server/
│   └── app.py                 # FastAPI server (/reset /step /state /health)
│
├── env/
│   ├── __init__.py
│   ├── environment.py         # ChronoVeritasEnv — main env class
│   ├── models.py              # Pydantic v2 typed models
│   ├── actions.py             # Action dispatcher
│   └── state_manager.py       # EpisodeState
│
├── graders/
│   ├── base_grader.py         # Shared helpers + all sub-component methods
│   ├── easy_grader.py         # 5 components
│   ├── medium_grader.py       # 7 components
│   └── hard_grader.py         # 9 components
│
├── search/
│   ├── bm25_index.py          # BM25 search
│   └── corpus_store.py        # Document store
│
├── agents/
│   ├── task_bank.py           # 15 SeedFact objects
│   ├── mutator.py             # Rule-based mutation engine
│   └── spreader.py            # Rule-based task constructor
│
├── training/
│   ├── sft_warmup.py          # Phase 0: SFT warm-start
│   └── reward_fn.py           # compute_reward() + extract_json_safe()
│
├── train_grpo.py              # Main GRPO training script
├── colab_training.ipynb       # Colab notebook version
│
├── eval/
│   ├── plot_results.py        # Generate all 4 required plots
│   └── qualitative_comparison.md  # Before/after examples
│
├── plots/                     # Committed .png files
│   ├── reward_curve_easy.png
│   ├── reward_curve_full_curriculum.png
│   ├── component_breakdown.png
│   └── baseline_vs_trained.png
│
├── training_logs/             # reward_log.json + reward_log.csv
│
└── data/
    └── tasks/
        ├── task_easy.json
        ├── task_medium.json
        ├── task_hard.json
        └── generated/         # Dynamically generated tasks
```

---

## 16. Risks & Mitigations

| Risk | Severity | Mitigation |
|---|---|---|
| GRPO reward stays flat (model never gets reward > 0) | High | SFT warm-start ensures model can produce valid JSON before GRPO. If still flat after 50 steps, lower temperature to 0.7 to reduce noise. |
| Model learns to pattern-match mutation type from claim wording | Medium | 4 mutation types × 15 seed facts × shuffled corpus order prevents simple pattern matching. Monitor if mutation_type reward improves before verdict — a sign of pattern gaming. |
| Reward component weights sum != 1.0 | Low | BaseGrader._validate_weights() logs a warning if positive weights deviate > 0.06 from 1.0. Check this at startup. |
| VRAM OOM during GRPO (8 generations × 256 tokens) | Medium | Reduce num_generations to 4. Reduce max_new_tokens to 128. Reduce per_device_train_batch_size to 1 (already 1). Enable gradient_checkpointing (already enabled). |
| Task generation fails (Mutator finds no pattern to apply) | Low | Mutator has fallback handlers for all 4 mutation types. If no pattern matches, it appends a generic mutation. Log warnings when fallbacks trigger. |
| Provenance chain not improving despite verdict improving | Medium | Reduce hallucination_penalty weight from 0.05 to 0.02 temporarily — this relaxes the constraint on provenance and lets the model focus on verdict first. Re-raise after Phase 1. |
| Judges want to see the environment in action, not just plots | Always | `inference.py --demo` should print a full episode trace to stdout with colour-coded rewards per step. Commit a sample trace to `eval/demo_trace.txt`. |
| Train–eval mismatch confuses judges | Medium | The README and blog explicitly state: "GRPO proxy trains reasoning; full grader measures strategy." Include a table matching proxy components to grader components. |

---

## 17. Submission Checklist

Before submitting, verify every item:

**Code:**
- [ ] `python train_grpo.py --difficulty easy --steps 50` runs without error
- [ ] `uvicorn server.app:app --port 7860` starts, `/health` returns 200
- [ ] `pytest` passes (no import errors at minimum)
- [ ] `openenv.yaml` validates with `openenv-core validate openenv.yaml`

**Evidence:**
- [ ] `plots/reward_curve_easy.png` — both axes labelled, reward goes up
- [ ] `plots/baseline_vs_trained.png` — shows quantitative improvement
- [ ] `training_logs/reward_log.csv` committed (judges can verify numbers)
- [ ] At least 1 qualitative example in `eval/qualitative_comparison.md`

**Framing (critical for storytelling score):**
- [ ] README does NOT say "multi-agent RL" — says "adversarial environment, trained Fact-Checker"
- [ ] README explicitly acknowledges train–eval delta in one paragraph
- [ ] README links to: HF Space, mini-blog/video, plots, Colab notebook
- [ ] Mini-blog answers: what capability gap, what the agent does, what improved, why it matters

**Deployment:**
- [ ] HF Space is live and `/health` returns 200 from the public URL
- [ ] Colab notebook runs end-to-end (test on a fresh Colab session)
- [ ] `Dockerfile` builds and runs with `docker run --cpus=2 --memory=8g`

---

*ChronoVeritas v2.1 · Shashank Tippanavar (IMT2022014) · OpenEnv Hackathon India 2026*
*Corrected framing: adversarial environment with single trained Fact-Checker. Mutator and Spreader are rule-based scripted components, not trained agents.*

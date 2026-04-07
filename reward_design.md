# ChronoVeritas — Reward Function Redesign

## The Core Problem with the Current System

The current `set_mutation_point` action acts as a **ground-truth oracle**:

```
Agent tries (DOC-A, distortion) → reward=0.20 → "one is right"
Agent tries (DOC-A, omission)   → reward=0.00 → "distortion was the correct type"
Agent tries (DOC-B, distortion) → reward=0.40 → "DOC-B is the correct doc"
```

A blind agent that never reads a single document can brute-force a perfect mutation point in `|docs| × |types|` = ~20 attempts. This means the reward function tests the agent's ability to do **combinatorial search**, not **fact-checking reasoning**.

### The Information Leakage Test

> *"Could an agent that ignores all document content and only reads reward signals achieve a non-trivial score?"*

If yes → the reward function leaks ground-truth information → broken.

Current system: **YES** — a doc-blind agent can get 0.40 from mutation point + whatever the grader gives for a lucky verdict guess. The reward is doing the reasoning *for* the agent.

---

## What Are Atomic Thought Rewards?

The concept comes from process reward models (PRMs) in LLM alignment research. The idea:

**Instead of rewarding WHAT the agent concludes, reward HOW it reasons — at each atomic step.**

An "atomic thought" in ChronoVeritas is a single investigative action:

| Atomic Thought | Action | What the agent is "thinking" |
|---|---|---|
| "I need to find documents about budget increases" | `search` | Query formulation |
| "This document looks relevant, let me read it" | `fetch_doc` | Relevance judgment |
| "This document says event X happened at time T" | `add_timeline_event` | Information extraction |
| "These two documents disagree about fact Y" | `flag_contradiction` | Contradiction detection |
| "I believe the mutation originated in this document" | `set_mutation_point` | Causal attribution |
| "My final verdict is..." | `submit_verdict` | Synthesis and judgment |

For each thought, we design a reward that measures **reasoning quality** without revealing **reasoning correctness**.

### The Teacher Analogy

Think of a math teacher grading a proof:
- ✅ Gives points for valid logical steps (even if the final answer is wrong)
- ✅ Gives points for showing work (evidence of reasoning)
- ❌ Does NOT say "your answer is close to 42" (that would leak the answer)
- ❌ Does NOT give points just because the final number is right (that rewards guessing)

---

## The Proposed Reward System

### Design Principles

1. **Zero answer leakage** — No reward signal should help the agent deduce the ground truth
2. **Process over outcome** — Reward the quality of investigation, not proximity to the answer
3. **Self-consistency** — Reward the agent for being internally coherent
4. **Evidence-grounding** — Reward the agent for basing conclusions on gathered evidence
5. **Diminishing returns** — Reward exploration early, penalize redundancy late

### Reward Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    EPISODE REWARD TIMELINE                       │
├─────────┬───────────────────────────┬───────────────────────────┤
│ Phase   │ Actions                   │ Reward Type               │
├─────────┼───────────────────────────┼───────────────────────────┤
│ Explore │ search, fetch_doc         │ Investigation Quality     │
│ Reason  │ timeline, contradiction   │ Reasoning Coherence       │
│ Declare │ set_mutation_point        │ Evidence Grounding (ONLY) │
│ Submit  │ submit_verdict            │ Full Terminal Grading     │
└─────────┴───────────────────────────┴───────────────────────────┘
```

---

### Signal 1: Search Specificity Reward

**When:** After each `search` action  
**Measures:** How targeted is the query vs. just dumping the entire claim?

**How it works:**

```
query_tokens = tokenize(query)
claim_tokens = tokenize(claim)

# Ratio of query tokens to claim tokens
specificity = 1.0 - len(query_tokens ∩ claim_tokens) / len(claim_tokens)
```

- Agent searches with the entire claim verbatim → `specificity ≈ 0.0` → reward = 0.00
- Agent searches with "budget transport council" → `specificity ≈ 0.6` → reward = +0.02
- Agent searches with "15% increase" (targeted) → `specificity ≈ 0.8` → reward = +0.03

**Why it's safe:** This measures query quality, not whether the query finds relevant documents. A blind agent gains nothing.

**What it teaches the agent:** Use focused, hypothesis-driven searches, not lazy copy-paste of the claim.

---

### Signal 2: Source Diversity Reward

**When:** After each `fetch_doc` action  
**Measures:** Is the agent reading diverse sources or just one type?

**How it works:**

```
sources_so_far = {doc.source for doc in fetched_docs}
new_source = (this_doc.source not in sources_so_far)

reward = 0.02 if new_source else 0.00
```

- First doc from "fda.gov" → +0.02 (new source type)
- Second doc from "fda.gov" → +0.00 (redundant)
- First doc from "healthnews.com" → +0.02 (new perspective)

**Why it's safe:** Rewards reading diverse perspectives, not specific documents. Doesn't indicate which source is "correct."

**What it teaches:** Good fact-checkers consult multiple source types (primary, secondary, commentary).

---

### Signal 3: Timeline Consistency Reward

**When:** After each `add_timeline_event` action  
**Measures:** Is the agent's timeline annotation consistent with the document's actual metadata?

**How it works:**

```
doc = get_fetched_doc(event.doc_id)

# 1. Timestamp consistency: does the agent's timestamp match the doc?
if event.timestamp == doc.timestamp:
    time_score = 1.0
elif event.timestamp is within ±1 day:
    time_score = 0.5
else:
    time_score = 0.0

# 2. Content grounding: does the event_label appear in the doc's content?
label_words = tokenize(event.event_label)
content_words = tokenize(doc.content)
grounding = len(label_words ∩ content_words) / len(label_words)

# 3. Ordering consistency: is the agent building the timeline in chronological order?
if len(timeline) >= 2:
    last_ts = timeline[-2].timestamp
    this_ts = timeline[-1].timestamp
    ordered = 1.0 if this_ts >= last_ts else 0.0
else:
    ordered = 1.0

reward = 0.01 × (time_score + grounding + ordered) / 3
```

Max reward per event: ~0.01. Over 5 events: ~0.05 total.

**Why it's safe:** Only checks if the agent correctly reads what's IN the document — not whether the document matters for the task. A blind agent that makes up timestamps scores 0.

**What it teaches:** Pay attention to what documents actually say. Build timelines carefully.

---

### Signal 4: Contradiction Quality Reward

**When:** After each `flag_contradiction` action  
**Measures:** Do the flagged documents actually contain textual disagreement?

**How it works:**

```
doc_a_content = fetched_docs[doc_id_a].content
doc_b_content = fetched_docs[doc_id_b].content

# Extract claim-relevant sentences from each doc
a_sentences = extract_claim_relevant(doc_a_content, claim)
b_sentences = extract_claim_relevant(doc_b_content, claim)

# Measure lexical divergence on claim-relevant content
shared_entities = named_entities(a_sentences) ∩ named_entities(b_sentences)
divergent_modifiers = modifiers_that_differ(a_sentences, b_sentences)

# High shared entities + high divergent modifiers = genuine contradiction
contradiction_score = len(shared_entities) × len(divergent_modifiers)

reward = 0.03 if contradiction_score > threshold else 0.00
```

Simpler approximation (no NER needed):

```
# Both docs mention similar topics but use different numbers/qualifiers
a_numbers = extract_numbers(doc_a_content)
b_numbers = extract_numbers(doc_b_content)

# Same topic (high keyword overlap) but different specifics (different numbers)
topic_overlap = jaccard(keywords(a), keywords(b))
fact_divergence = 1.0 - jaccard(a_numbers, b_numbers)

reward = 0.03 if topic_overlap > 0.3 and fact_divergence > 0.3 else 0.00
```

**Why it's safe:** Measures whether the agent found a *real* contradiction, not whether it found the *right* contradiction. Two documents about the same topic with different numbers IS a real contradiction — regardless of which one the ground truth considers "correct."

**What it teaches:** Flag genuine disagreements between documents, not arbitrary pairs.

---

### Signal 5: Evidence-Grounding Reward (replaces current `set_mutation_point` reward)

**When:** After `set_mutation_point`  
**Measures:** How well-supported is the agent's declaration by its own investigation?  
**CRITICAL: Does NOT compare against ground truth AT ALL**

**How it works:**

```
declared_doc = payload.doc_id
declared_type = payload.mutation_type

# 1. Did the agent actually read this document?
read_score = 1.0 if declared_doc in fetched_doc_ids else 0.0

# 2. Did the agent flag a contradiction involving this document?  
contradiction_score = 1.0 if declared_doc in any(contradictions) else 0.0

# 3. Did the agent add a timeline event for this document?
timeline_score = 1.0 if declared_doc in [e.doc_id for e in timeline] else 0.0

# 4. Is the declaration consistent with the agent's own contradiction flags?
# (If you flagged A vs B as contradictory, declaring A or B makes sense)
consistency_score = 1.0 if declared_doc in flagged_contradiction_docs else 0.5

# Combined evidence-grounding score
grounding = (read_score + contradiction_score + timeline_score + consistency_score) / 4

reward = 0.05 × grounding
```

Possible rewards:
- Agent fetched the doc, flagged it in a contradiction, has a timeline event, and it's consistent → **0.05**
- Agent fetched the doc but did nothing else with it → **0.0125**
- Agent declares a doc it never fetched → **0.00**

**Why it's safe:** Notice there is ZERO comparison against ground truth. The reward only asks: "Did you do your homework on this document before pointing at it?" A blind probing agent gets 0.00 every time because it never fetches or annotates documents.

**What it teaches:** Don't declare a mutation point you haven't investigated. Build evidence before making claims.

---

### Signal 6: Terminal Grading (unchanged)

**When:** After `submit_verdict`  
**Measures:** Correctness of the final answer (all components)

This stays exactly as it is now — the EasyGrader/MediumGrader/HardGrader system. This is the only place where ground truth comparison happens.

---

## Complete Reward Table (Proposed vs Current)

| Action | Current Reward | Proposed Reward | Leaks Answer? |
|--------|---------------|-----------------|---------------|
| `search` | 0.00 | 0.00–0.03 (query specificity) | ❌ No |
| `fetch_doc` | 0.00 | 0.00–0.02 (source diversity) | ❌ No |
| `add_timeline_event` | 0.00 | 0.00–0.01 (consistency) | ❌ No |
| `flag_contradiction` | 0.00 | 0.00–0.03 (contradiction quality) | ❌ No |
| `set_mutation_point` | **0.00–0.40 (GT comparison!)** | 0.00–0.05 (evidence grounding) | ❌ No → ✅ Fixed |
| `submit_verdict` | Full grading (0.0–1.0) | Full grading (0.0–1.0) | N/A (terminal) |

**Maximum intermediate reward:** ~0.15 (across all non-terminal actions)  
**Maximum terminal reward:** 1.0 (from grader)  
**Ratio:** Intermediate signals are ~15% of max score — enough to shape behavior, not enough to dominate scoring.

---

## How the Agent Strategy Changes

### Current Agent (v5) — Probe and Exploit

```
1. Search → Fetch all docs
2. Ask LLM for answer
3. Try set_mutation_point with LLM's guess
4. If reward=0.20 → PROBE: same doc, different types (binary search)
5. If reward=0.00 → try next doc candidate
6. Submit whatever combination got reward=0.40
```

This agent's intelligence is in the **probing loop** (Steps 3-5), not in the LLM reasoning.

### Proposed Agent — Reason and Justify

```
1. Search with targeted queries (rewarded for specificity)
2. Fetch diverse sources (rewarded for source variety)
3. Build timeline from each doc (rewarded for consistency)
4. Flag genuine contradictions (rewarded for quality)
5. Ask LLM: given all evidence, what's the verdict?
6. Declare mutation point (rewarded ONLY for evidence-grounding)
7. Submit verdict (graded on correctness)
```

This agent's intelligence must be in **Steps 3-5** — the actual investigative reasoning. There's no reward oracle to probe. The only way to score well on the terminal grading is to actually reason correctly about the documents.

### Key Behavioral Difference

| Behavior | Current System | Proposed System |
|----------|---------------|-----------------|
| Reading documents | Optional (can probe without reading) | Essential (evidence-grounding rewards) |
| Multiple mutation attempts | Encouraged (binary search) | Pointless (no GT comparison) |
| Building timeline | Cosmetic (no reward) | Valuable (+0.01 per consistent event) |
| Flagging contradictions | Cosmetic (no reward) | Valuable (+0.03 for genuine contradictions) |
| LLM reasoning quality | Irrelevant (probing overrides LLM) | Critical (only way to get terminal score) |

---

## Reward Shaping Theory: Why This Is Sound

### Potential-Based Reward Shaping (Ng et al., 1999)

The theory says: you can add intermediate rewards to an MDP without changing the optimal policy, **as long as** the shaped reward is derived from a potential function of the state (not the action or transition).

Our proposed rewards satisfy this because they are functions of **observable state** (what the agent has fetched, annotated, flagged) — not of the hidden ground truth. The optimal policy under shaped rewards is the same as under terminal-only rewards: investigate thoroughly, then reason correctly.

### The Reward Components Form a DAG

```
search_specificity ──→ triggers better search results
        │                        │
        ▼                        ▼
source_diversity ──→ broader evidence base
        │                        │
        ▼                        ▼
timeline_consistency ──→ structured understanding
        │                        │
        ▼                        ▼
contradiction_quality ──→ identifies key disagreements
        │                        │
        ▼                        ▼
evidence_grounding ──→ well-supported declaration
        │                        │
        ▼                        ▼
terminal_grading ──→ correct final answer (ONLY GT comparison)
```

Each intermediate reward naturally leads to the next phase of investigation. An agent that maximizes intermediate rewards will naturally build a strong evidence base for its final answer — without ever being told what the answer is.

---

## Impact on the Grading System

### No changes needed to the graders themselves

The EasyGrader/MediumGrader/HardGrader classes remain unchanged. They still score verdict, mutation_type, mutation_point, provenance, timeline, efficiency, and hallucination at submit time.

### The only changes are in the action dispatcher

- `_handle_search` → add specificity bonus
- `_handle_fetch_doc` → add diversity bonus
- `_handle_add_timeline_event` → add consistency bonus
- `_handle_flag_contradiction` → add quality bonus
- `_handle_set_mutation_point` → **REMOVE GT comparison**, add evidence-grounding bonus
- `_handle_submit_verdict` → unchanged

### The inference agent needs a different strategy

- Remove the entire isolation probing loop
- Focus on thorough investigation: diverse searches, reading all docs, building proper timelines
- Trust the LLM's reasoning over the reward signal (because the reward no longer leaks the answer)
- Single `set_mutation_point` call (no probing), then immediate `submit_verdict`

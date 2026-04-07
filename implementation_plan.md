# ChronoVeritas v2 — Complete Environment & Reward Redesign

## Goal

Redesign ChronoVeritas from an environment with answer-leaking intermediate rewards into a proper RL evaluation environment with **terminal-only correctness grading** + **non-exploitable process bonuses**. This covers every layer: data model, actions, graders, and inference agent.

---

## 1. Data Model Changes

### 1.1 Add `reliability_tier` to Document/DocMeta

Every document gets a source reliability tier — visible to the agent at search time.

```
Tier 1 (Official)    — Government, academic, peer-reviewed, court records
Tier 2 (Institutional) — Major news orgs, press releases, corporate statements
Tier 3 (Informal)    — Blogs, forums, social media, unverified sources
```

This field goes into the task JSON corpus:

```json
{
  "doc_id": "DOC-0001",
  "title": "Council Minutes Jan",
  "source": "citycouncil.gov",
  "reliability_tier": 1,       ← NEW
  "timestamp": 1672531200,
  ...
}
```

The `DocMeta` and `Document` Pydantic models gain `reliability_tier: int = Field(ge=1, le=3, default=2)`.

**Why:** Source reliability is fundamental to fact-checking. Agents should learn to anchor on Tier-1 sources and treat Tier-3 with skepticism. The grader rewards this behavior through the `source_reliability_score` component.

### 1.2 Add `early_detection_achieved` to EpisodeState

A boolean flag set to `True` during the episode if the agent correctly identifies the mutation point within the first 40% of the step budget.

```
EpisodeState:
    early_detection_achieved: bool = False      ← NEW
    early_detection_step_pct: float = 1.0       ← NEW (records when it happened)
```

**Why:** Rewards agents that reach the correct hypothesis early rather than exhausting all steps. Checked only at terminal grading time — not leaked as intermediate reward.

### 1.3 Existing task data updates

All three task files (`task_easy.json`, `task_medium.json`, `task_hard.json`) need `reliability_tier` added to each corpus document. Here's the tier assignment for existing tasks:

**EASY-001 (Budget distortion):**

| Doc | Source | Tier |
|-----|--------|------|
| DOC-0001 | citycouncil.gov | 1 |
| DOC-0002 | citynews.com | 2 |
| DOC-0003 | metrodaily.com | 2 |
| DOC-0004 | weeklydigest.net | 3 |

**MED-001 (Drug recall omission):**

| Doc | Source | Tier |
|-----|--------|------|
| DOC-1001 | fda.gov | 1 |
| DOC-1002 | pharmaco.com | 2 |
| DOC-1003 | healthnewsdaily.com | 2 |
| DOC-1004 | medjournal.org | 1 |
| DOC-1005 | patientforum.net | 3 |

**HARD-001 (Research fabrication):**

| Doc | Source | Tier |
|-----|--------|------|
| DOC-2001 | university.edu | 1 |
| DOC-2002 | sciencedaily.com | 2 |
| DOC-2003 | viralhealth.blog | 3 |
| DOC-2004 | university.edu | 1 |
| DOC-2005 | factcheck.org | 2 |

---

## 2. Action Dispatcher Changes

### 2.1 Action reward table (new vs current)

| Action | Step Cost | Current Reward | New Reward | Change |
|--------|-----------|---------------|------------|--------|
| `search` | 1 step | 0.0 | 0.0 | No change |
| `fetch_doc` | 1 step + tokens | 0.0 | 0.0 | No change |
| `add_timeline_event` | 0 (free) | 0.0 | +0.02 max (consistency) | New process bonus |
| `flag_contradiction` | 0 (free) | 0.0 | +0.02 max (quality) | New process bonus |
| `set_mutation_point` | 0 (free) | **0.0–0.40 (GT comparison!)** | **0.0 (just records)** | **GT leak removed** |
| `submit_verdict` | 0 (terminal) | Full grading | Full grading (enhanced) | Enhanced components |

### 2.2 `search` — No reward change

Remains at 0.0. The step cost is the implicit penalty for wasteful searches.

The search handler itself stays the same: BM25 query → returns ranked DocMeta results → adds newly discovered docs to state corpus.

> The key fix from earlier still applies: reset gives empty corpus, search is the only way to discover documents.

### 2.3 `fetch_doc` — No reward change

Remains at 0.0. Step cost + token budget deduction are the implicit costs.

The discovery-first check (from our earlier fix) stays: agent must have found the doc via search before fetching.

### 2.4 `add_timeline_event` — Process bonus: Timestamp Consistency

**Reward logic (max +0.02 per event):**

```
Step 1: Check if the agent actually fetched this document
  → fetched = doc_id in state.fetched_doc_ids
  → if not fetched: reward = 0.0 (can't verify consistency)

Step 2: Check timestamp consistency
  → doc = state.get_fetched_doc(doc_id)
  → if event.timestamp == doc.timestamp: time_score = 1.0
  → elif event.timestamp is None: time_score = 0.0 (agent didn't commit — no free points)
  → else: time_score = 0.0 (wrong timestamp)

Step 3: Check event label grounding
  → label_words = tokenize(event_label)
  → content_words = tokenize(doc.content)
  → overlap = |label_words ∩ content_words| / |label_words|
  → grounding = min(overlap, 1.0)

Final reward = 0.02 × (time_score + grounding) / 2
```

**Why this doesn't leak the answer:** It only checks if the agent correctly reads what's IN the document — not whether the document matters for the task. An agent that makes up timestamps or labels scores 0.

**Cap:** Maximum total timeline bonus across all events = 0.10 (cap at 5 events rewarded). This prevents gaming by spamming timeline events.

### 2.5 `flag_contradiction` — Process bonus: Contradiction Quality

**Reward logic (max +0.02 per flag):**

```
Step 1: Check if both documents were fetched
  → if doc_id_a not in fetched_doc_ids or doc_id_b not in fetched_doc_ids:
      reward = 0.0 (can't flag what you haven't read)

Step 2: Both docs fetched → reward
  → reward = 0.02
```

Simple evidence-grounding check: you can only flag a contradiction between documents you've actually read. No textual analysis of content — that adds complexity without proportional value.

**Why this doesn't leak the answer:** Only checks whether the agent did its homework (read both docs). Doesn't evaluate whether the contradiction is "correct" or relevant to the ground truth.

**Cap:** Maximum total contradiction bonus = 0.04 (cap at 2 flags rewarded).

### 2.6 `set_mutation_point` — CRITICAL CHANGE: Remove GT comparison

**Current behavior (BROKEN):**
```
if doc_id == gt.gt_mutation_doc_id: reward += 0.20
if mutation_type == gt.gt_mutation_type: reward += 0.20
state.record_reward(reward)
```

**New behavior:**
```
Step 1: Record the declaration on state
  → state.declared_mutation = MutationDecl(doc_id, mutation_type)

Step 2: Compute evidence-grounding score (NO GT comparison)
  → read_it = 1.0 if doc_id in state.fetched_doc_ids else 0.0
  → flagged_it = 1.0 if doc_id appears in any contradiction pair else 0.0
  → timeline_it = 1.0 if doc_id appears in any timeline event else 0.0
  → grounding = (read_it + flagged_it + timeline_it) / 3

Step 3: Return observation feedback (not reward)
  → info["evidence_grounding"] = grounding
  → info["evidence_details"] = {
        "document_fetched": read_it > 0,
        "contradiction_flagged": flagged_it > 0,
        "timeline_annotated": timeline_it > 0,
    }

Step 4: Check early detection (stored for terminal grading, NOT rewarded now)
  → steps_used_pct = state.current_step / state.max_steps
  → Compare against GT (silently — result not shown to agent):
      if doc_id == gt.gt_mutation_doc_id and mutation_type == gt.gt_mutation_type:
          if steps_used_pct <= 0.40:
              state.early_detection_achieved = True
              state.early_detection_step_pct = steps_used_pct

Step 5: Return
  → reward = 0.0  (ALWAYS zero — no answer leakage)
  → done = False
```

**What the agent sees:** Only whether its own evidence base supports its declaration — never whether the declaration is correct.

**Why the early detection check is safe:** The result is stored silently on state and only used at terminal grading inside `submit_verdict`. The agent never sees `state.early_detection_achieved` in any observation or reward signal during the episode.

### 2.7 `submit_verdict` — Unchanged dispatcher, enhanced grader

The dispatcher behavior stays the same:
1. Validate `VerdictPayload` via Pydantic
2. Transition phase to `GRADING`
3. Return `(0.0, {_verdict_obj: verdict}, done=True)`

The grading (in `environment.step()`) calls the enhanced grader — see Section 3.

---

## 3. Enhanced Grading System

### 3.1 New scoring components

The grading system expands from 7 to 10 components:

| # | Component | Range | Description |
|---|-----------|-------|-------------|
| 1 | `verdict_accuracy` | {0, 1} | Exact match on verdict string |
| 2 | `source_reliability` | [0, 1] | **NEW** — Weighted avg tier of docs in provenance chain |
| 3 | `mutation_type_score` | {0, 1} | Exact match on mutation type |
| 4 | `mutation_point_score` | {0, 0.5, 1} | Doc match (0.5 if adjacent in timeline) |
| 5 | `provenance_chain` | [0, 1] | **ENHANCED** — F1 + direction + origin tier |
| 6 | `timeline_score` | [0, 1] | Kendall-tau on timeline ordering |
| 7 | `efficiency_score` | [0, 1] | 1 − steps_used / max_steps |
| 8 | `early_detection_bonus` | {0, 1} | **NEW** — Correct mutation before 40% budget |
| 9 | `reconciliation_score` | [0, 1] | **NEW** — Hard only: multi-source field matching |
| 10a | `hallucination_penalty` | [0, 1] | Penalty: fabricated/unread doc citations |
| 10b | `brier_penalty` | [0, 1] | **NEW** — Penalty: miscalibrated confidence |

### 3.2 Component: `source_reliability` (NEW)

**What it measures:** Did the agent anchor its provenance chain to authoritative sources?

```
TIER_WEIGHTS = {1: 1.0, 2: 0.6, 3: 0.2}

For each doc_id in agent's provenance_chain:
    look up reliability_tier from corpus
    map to weight

source_reliability = average(weights)
```

**Examples:**

| Agent's chain | Tiers | Score |
|---|---|---|
| [DOC-0001 (T1), DOC-0003 (T2)] | [1.0, 0.6] | **0.80** |
| [DOC-0003 (T2), DOC-0004 (T3)] | [0.6, 0.2] | **0.40** |
| [DOC-0004 (T3)] | [0.2] | **0.20** |

**Why it matters:** An agent that builds its case from official sources (FDA notices, council minutes) demonstrates better fact-checking methodology than one relying on blogs and forums.

### 3.3 Component: `provenance_chain` (ENHANCED — 3 sub-scores)

**Current:** Simple multiset F1 (unordered comparison).

**Enhanced:** 3 sub-components capturing different aspects:

#### Sub-score A: Membership F1 (30% weight)

Unchanged from current — multiset-aware F1 between predicted and GT provenance chains.

```
pred_counts = Counter(agent_chain)
gt_counts = Counter(gt_chain)
overlap = Σ min(pred[k], gt[k]) for shared keys
precision = overlap / Σpred_counts
recall = overlap / Σgt_counts
f1 = 2⋅precision⋅recall / (precision + recall)
```

#### Sub-score B: Directional Ordering (40% weight)

**Why 40%?** Claim mutation is a causal process — A caused B caused C. The ORDER matters more than the membership. An agent that submits [DOC-0003, DOC-0001] (reversed) misunderstands the causal flow even though it found the right documents.

```
common = [d for d in agent_chain if d in gt_set]
if len(common) >= 2:
    agent_ranks = [agent_chain.index(d) for d in common]
    gt_ranks = [gt_chain.index(d) for d in common]
    tau, _ = kendalltau(agent_ranks, gt_ranks)
    direction_score = (tau + 1) / 2      # normalise [-1,1] → [0,1]
else:
    direction_score = 0.5  # insufficient data → neutral
```

**Examples for EASY-001** (GT chain: [DOC-0001, DOC-0003]):

| Agent chain | Direction | Score |
|---|---|---|
| [DOC-0001, DOC-0003] | Correct order | **1.00** |
| [DOC-0003, DOC-0001] | Reversed | **0.00** |
| [DOC-0001, DOC-0002, DOC-0003] | Correct (superset) | **1.00** |
| [DOC-0003, DOC-0002, DOC-0001] | Fully reversed | **0.00** |

#### Sub-score C: Origin Source Tier (30% weight)

**What it measures:** Does the first document in the agent's chain start from a Tier-1 source?

```
if agent_chain is empty:
    tier_score = 0.0
else:
    origin_doc = agent_chain[0]
    origin_tier = corpus[origin_doc].reliability_tier
    tier_score = {1: 1.0, 2: 0.6, 3: 0.2}[origin_tier]
```

**Why:** A provenance chain should start from the authoritative original source and trace forward to the mutation. Starting from a blog (Tier 3) suggests the agent doesn't understand provenance direction.

#### Combined provenance score:

```
provenance_chain_score = 0.30 × f1 + 0.40 × direction + 0.30 × origin_tier
```

### 3.4 Component: `early_detection_bonus` (NEW)

**What it measures:** Did the agent identify the correct mutation point within the first 40% of its step budget?

```
early_detection_bonus = 1.0 if state.early_detection_achieved else 0.0
```

This flag was set silently during `set_mutation_point` (Section 2.6).

**Why:** Rewards efficient reasoning. An agent that figures out the mutation early demonstrates stronger analytical capability. The bonus is small (0.05 weight) — it's a tiebreaker, not a dominant factor.

**Why it's safe:** The agent never sees this flag during the episode. It's only checked at terminal grading time.

### 3.5 Component: `reconciliation_score` (NEW — Hard only)

**What it measures:** How well the agent reconciles conflicting information across multiple sources.

This is the most complex new component, active only for hard tasks. It checks whether the agent identified the key discrepancies:

```
Step 1: Identify the "conflict fields" in the task
  → Each hard task defines gt_conflict_fields in ground_truth:
    e.g., ["species_studied", "causation_claim", "peer_review_status"]

Step 2: For each conflict field, check if the agent's investigation covers it
  → Does the agent's timeline/contradiction set show awareness of this conflict?
  → Heuristic: did the agent flag a contradiction between docs that disagree on this field?

Step 3: Score
  → fields_covered = count of conflict fields where agent flagged relevant contradiction
  → reconciliation = fields_covered / total_conflict_fields
```

**Example for HARD-001** (conflict fields: species, causation, peer review):

| Agent flagged contradictions | Fields covered | Score |
|---|---|---|
| DOC-2001 vs DOC-2003 (species: zebrafish vs human) | 1/3 | **0.33** |
| DOC-2001 vs DOC-2003 + DOC-2003 vs DOC-2004 (species + peer review) | 2/3 | **0.67** |
| All three conflict points covered | 3/3 | **1.00** |

> [!NOTE]
> For v1 implementation, this can be simplified: `reconciliation = min(len(state.contradictions), len(gt_conflict_fields)) / len(gt_conflict_fields)`. The agent gets credit for flagging more contradictions (up to the number of actual conflict fields), without the system revealing WHICH contradictions are correct.

### 3.6 Component: `brier_penalty` (NEW)

**What it measures:** Is the agent's confidence calibrated with the complexity of the task and correctness of its answer?

```
Step 1: Compute temporal decay from chain length
  chain_len = len(gt_provenance_chain)
  decay = max(0.5, 1.0 - (chain_len - 2) × 0.1)

  Chain of 2 docs: decay = 1.0 (simple case, high confidence expected)
  Chain of 4 docs: decay = 0.8 (moderate complexity)
  Chain of 6 docs: decay = 0.6 (complex chain, uncertainty expected)

Step 2: Compute expected confidence
  correctness = 1.0 if agent_verdict == gt_verdict else 0.0
  expected_conf = correctness × decay

Step 3: Brier score
  brier = (agent_confidence − expected_conf)²
  brier_penalty = min(brier, 1.0)
```

**Worked examples:**

| Scenario | Verdict correct? | Chain len | Decay | Agent conf | Expected | Brier |
|---|---|---|---|---|---|---|
| Easy, right, confident | ✅ | 2 | 1.0 | 0.95 | 1.0 | **(0.95−1.0)² = 0.0025** |
| Easy, right, underconfident | ✅ | 2 | 1.0 | 0.50 | 1.0 | **(0.50−1.0)² = 0.2500** |
| Hard, right, overconfident | ✅ | 4 | 0.8 | 0.95 | 0.8 | **(0.95−0.8)² = 0.0225** |
| Hard, right, calibrated | ✅ | 4 | 0.8 | 0.75 | 0.8 | **(0.75−0.8)² = 0.0025** |
| Easy, wrong, confident | ❌ | 2 | 1.0 | 0.90 | 0.0 | **(0.90−0.0)² = 0.8100** |
| Easy, wrong, uncertain | ❌ | 2 | 1.0 | 0.20 | 0.0 | **(0.20−0.0)² = 0.0400** |

**Key insight:** This penalises both overconfidence on wrong answers AND underconfidence on right answers, while accounting for task complexity.

---

## 4. Weight Tables — All Three Tiers

### Full weight assignment:

| Component | Easy | Medium | Hard | Notes |
|---|---|---|---|---|
| `verdict_accuracy` | **0.45** | **0.25** | **0.20** | Core — always graded |
| `source_reliability` | **0.10** | **0.10** | **0.08** | Tier-weighted avg of chain |
| `mutation_type_score` | 0.00 | **0.20** | **0.10** | Easy doesn't test type |
| `mutation_point_score` | **0.35** | **0.20** | **0.10** | Higher on easy — it's the key alpha |
| `provenance_chain` | **0.05** | **0.10** | **0.15** | F1 + direction + origin tier |
| `timeline_score` | 0.00 | 0.00 | **0.07** | Kendall tau — hard only |
| `efficiency_score` | 0.00 | **0.10** | **0.05** | 1 − steps/max_steps |
| `early_detection_bonus` | **0.05** | **0.05** | **0.05** | Correct mutation < 40% budget |
| `reconciliation_score` | 0.00 | 0.00 | **0.15** | Hard only — field-by-field |
| *Sum of positives* | *1.00* | *1.00* | *0.95* | |
| `hallucination_penalty` | −0.10 | −0.10 | −0.15 | 0.05 per unfetched, 0.10 per fabricated |
| `brier_penalty` | −0.05 | −0.08 | −0.10 | (conf − temporal_expected)² |

### Why Hard positives sum to 0.95

The reconciliation_score (0.15) makes a perfect hard score ~0.95 before penalties. This is deliberate:
- Hard tasks should never trivially reach 1.0
- It acts as a ceiling stretch goal — forces mastery across ALL components
- After potential penalties (up to −0.25), hard baseline scores are 0.15–0.30

### Target baseline scores by difficulty:

| Difficulty | Random agent | Decent agent | Expert agent |
|---|---|---|---|
| Easy | 0.10–0.15 | 0.55–0.70 | 0.85–0.95 |
| Medium | 0.05–0.10 | 0.35–0.50 | 0.75–0.90 |
| Hard | 0.02–0.05 | 0.15–0.30 | 0.55–0.75 |

---

## 5. Complete Reward Flow — One Episode

### Step-by-step reward accumulation:

```
reset("EASY-001")
  → Agent sees: claim only, corpus_metadata = [], max_steps = 15
  → Reward: 0.0

search("budget transport council")                        [step 1]
  → Agent discovers DOC-0001, DOC-0002, DOC-0003, DOC-0004
  → Reward: 0.0

fetch_doc("DOC-0001")                                     [step 2, −34 tokens]
  → Agent reads council minutes (Tier 1)
  → Reward: 0.0

fetch_doc("DOC-0003")                                     [step 3, −27 tokens]
  → Agent reads Metro Daily (Tier 2) — sees "15%" claim
  → Reward: 0.0

fetch_doc("DOC-0002")                                     [step 4, −21 tokens]
  → Agent reads city news (Tier 2) — generic, no percentage
  → Reward: 0.0

add_timeline_event(DOC-0001, "Council voted 5% increase", ts=1672531200)
  → timestamp matches doc ✓, "council" + "voted" in content ✓
  → Reward: +0.02

add_timeline_event(DOC-0003, "Metro reports 15% increase", ts=1672704000)
  → timestamp matches ✓, "metro" + "15%" in content ✓
  → Reward: +0.02

flag_contradiction(DOC-0001, DOC-0003)
  → Both fetched ✓
  → Topic overlap: "budget", "increase", "council" shared (Jaccard ~0.4) ✓
  → Number divergence: "5%" vs "15%" (different) ✓
  → Quality > 0.3 ✓
  → Reward: +0.02

set_mutation_point(DOC-0003, distortion)
  → Evidence grounding:
      document_fetched: ✓
      contradiction_flagged: ✓
      timeline_annotated: ✓
  → Silently checks GT: DOC-0003 matches, distortion matches → early_detection = True
  → Reward: 0.0 (agent sees grounding feedback, NOT correctness)

submit_verdict(verdict="false", mutation_type="distortion",
               mutation_doc_id="DOC-0003", chain=["DOC-0001","DOC-0003"],
               confidence=0.90)
  → Terminal grading fires:
      verdict_accuracy:     "false"=="false"       → 1.0 × 0.45 = 0.450
      source_reliability:   [T1, T2] avg          → 0.8 × 0.10 = 0.080
      mutation_point:       DOC-0003 exact match   → 1.0 × 0.35 = 0.350
      provenance_chain:     F1=1.0, dir=1.0, tier=1.0 → 1.0 × 0.05 = 0.050
      early_detection:      achieved at 27% budget → 1.0 × 0.05 = 0.050
      hallucination:        no fabricated docs     → 0.0 × 0.10 = 0.000
      brier:                (0.90−1.0)²=0.01      → 0.01 × 0.05 = 0.001
      ─────────────────────────────────────────────────────────────
      TOTAL = 0.980 − 0.001 = 0.979

  → Reward: 0.979
```

**Total episode rewards:** [0, 0, 0, 0, +0.02, +0.02, +0.02, 0, 0.979]  
**Final score (from info.final_score):** 0.979

---

## 6. Inference Agent Strategy

### 6.1 Overview: Reason-and-Justify (replaces Probe-and-Exploit)

The agent can no longer brute-force the mutation point because `set_mutation_point` returns 0 reward. The optimal strategy is now:

```
Phase 1: Investigate (searches + fetches)
  → Use targeted queries to discover relevant documents
  → Prioritize fetching Tier-1 sources first
  → Aim for diverse source coverage

Phase 2: Annotate (timeline + contradictions)
  → Build chronological timeline from document timestamps
  → Flag contradictions between docs with conflicting facts
  → These are free AND give small process bonuses

Phase 3: Reason (LLM analysis)
  → Send all evidence to LLM with structured prompt
  → Ask for verdict, mutation type, mutation doc, provenance chain
  → Request calibrated confidence (teach LLM about Brier penalty)

Phase 4: Declare (single mutation point + immediate submit)
  → ONE set_mutation_point call (no probing)
  → Immediately submit_verdict
  → No isolation loop, no retry logic
```

### 6.2 Key strategy changes from v5

| v5 (current) | v2 (proposed) |
|---|---|
| Probes set_mutation_point 3-5 times | Calls set_mutation_point exactly once |
| Uses reward signal to binary-search answer | Ignores intermediate reward (always 0) |
| LLM reasoning is overridden by probing | LLM reasoning IS the answer |
| Fetches docs in any order | Prioritizes Tier-1 sources first |
| Timeline/contradictions are cosmetic | Timeline/contradictions earn process bonuses |
| Confidence is arbitrary | Confidence is calibrated (Brier penalty) |

### 6.3 System prompt additions for the agent

The agent's LLM prompt should add:

```
══ CONFIDENCE CALIBRATION ══════════════════════════════════════════
  Your confidence should reflect the complexity of the evidence chain.
  - Simple 2-doc contradiction: confidence 0.85–0.95
  - 3-4 doc chain with clear mutation: confidence 0.70–0.85
  - 5+ doc chain with subtle omission: confidence 0.55–0.70
  
  NEVER say confidence > 0.90 for long chains.
  NEVER say confidence < 0.50 if you've read multiple corroborating sources.

══ SOURCE HIERARCHY ═════════════════════════════════════════════════
  Tier 1 (Official): Government, academic, peer-reviewed — treat as ground truth
  Tier 2 (Institutional): Major news, press releases — generally reliable
  Tier 3 (Informal): Blogs, forums, social media — may be the mutation source
  
  Start your provenance chain from a Tier-1 source when possible.
  The mutation point is most likely in a Tier-2 or Tier-3 document.
```

### 6.4 Step budget allocation strategy

For an agent with N max steps:

| Phase | Steps allocated | Purpose |
|---|---|---|
| Search | 1–3 steps (7–20%) | Discover corpus |
| Fetch | 3–5 steps (20–33%) | Read key documents |
| Annotate | 0 steps (free) | Timeline + contradictions |
| LLM analysis | 0 steps (agent-side) | Reasoning |
| Declare + Submit | 0 steps (free + terminal) | Answer |
| **Reserve** | N × 0.40 steps | Reach early detection threshold |

The early detection bonus rewards finishing the investigation in ≤40% of the budget, so the agent should aim to complete search+fetch within the first 6 steps of a 15-step budget (40%).

---

## 7. Grader Architecture Changes

### 7.1 BaseGrader additions

New methods on BaseGrader:

```
_grade_source_reliability(provenance_chain, corpus) → float [0,1]
  → Weighted average of reliability_tier for docs in agent's chain

_grade_provenance_enhanced(agent_chain) → float [0,1]
  → 0.30 × membership_f1 + 0.40 × directional_ordering + 0.30 × origin_tier

_grade_early_detection(state) → float {0,1}
  → 1.0 if state.early_detection_achieved else 0.0

_grade_brier_penalty(confidence, verdict, gt_chain_length) → float [0,1]
  → (confidence − expected_confidence)², with temporal decay
```

### 7.2 HardGrader additions

New method specific to HardGrader:

```
_grade_reconciliation(state) → float [0,1]
  → Score based on how many conflict fields the agent covered
  → Simplified v1: min(len(contradictions), len(gt_conflict_fields)) / len(gt_conflict_fields)
```

### 7.3 Ground truth model additions

The `GroundTruth` model needs one new optional field:

```
class GroundTruth(BaseModel):
    gt_verdict: VerdictType
    gt_mutation_type: MutationType
    gt_mutation_doc_id: Optional[str]
    gt_provenance_chain: List[str]
    gt_timeline: List[str]
    gt_conflict_fields: List[str] = []    ← NEW (hard tasks only)
```

---

## 8. Summary of All Changes

| Layer | File | Change |
|---|---|---|
| **Models** | `env/models.py` | Add `reliability_tier` to DocMeta/Document, add `gt_conflict_fields` to GroundTruth |
| **State** | `env/state_manager.py` | Add `early_detection_achieved`, `early_detection_step_pct` |
| **Actions** | `env/actions.py` | Process bonuses for timeline/contradiction, remove GT leak from set_mutation_point, add silent early detection check |
| **Environment** | `env/environment.py` | Already fixed (empty corpus on reset) |
| **Graders** | `graders/base_grader.py` | Add `_grade_source_reliability`, `_grade_provenance_enhanced`, `_grade_early_detection`, `_grade_brier_penalty` |
| **Graders** | `graders/easy_grader.py` | New weight table, add source_reliability + early_detection + brier |
| **Graders** | `graders/medium_grader.py` | New weight table, add source_reliability + early_detection + brier |
| **Graders** | `graders/hard_grader.py` | New weight table, add all new components incl. reconciliation |
| **Data** | `data/tasks/*.json` | Add `reliability_tier` to corpus docs, add `gt_conflict_fields` to hard GT |
| **Inference** | `inference.py` | Remove probing loop, add source-priority fetching, add confidence calibration |

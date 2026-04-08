---
title: ChronoVeritas
emoji: 🔍
colorFrom: indigo
colorTo: purple
sdk: docker
pinned: false
short_description: An RL environment for temporal fact-checking & claim provenance
---

# ChronoVeritas — Claim Lifecycle Verification Environment

<p align="center">
  <strong>Can an AI trace how a fact becomes a falsehood?</strong><br>
  <em>Training RL agents to do what thousands of human fact-checkers do today — faster, cheaper, and without hallucinating.</em>
</p>

An [OpenEnv](https://openenv.ai)-compliant reinforcement learning environment for **temporal fact-checking**. Agents investigate how a factual claim mutates as it propagates through a document corpus — from authoritative primary sources through news reports to informal media — and must identify *where* the truth was distorted, *how* it was altered, and *why* the mutation matters.

## Motivation

Every major platform — Meta, X/Twitter, YouTube, Reuters — employs teams of humans to verify claims that spread online. They don't just need to know if something is true *today*. They need to know **when it became false, who distorted it, and where it originated.** This is a task with hundreds of millions of dollars of real-world labor behind it and **no scalable automated solution today.**

### The Problem at Scale

| Who | What They Do Today | Cost / Speed |
|---|---|---|
| **Meta / X / YouTube** | Human moderators flag viral posts that are distorted versions of real stories before they spread | **$200M+/year** in content moderation spend |
| **Reuters / AP / Bloomberg** | Journalists cross-reference breaking claims against archived sources before publishing | **30–90 minutes** manually per story |
| **SEC / Financial compliance** | Teams detect when an executive's current statement contradicts a prior filing — a legal requirement | **Hundreds of analysts**, mandatory under law |

Existing AI approaches fall short. X's Grok chatbot has a documented history of hallucinating facts and generating incorrect information — it can *produce* misinformation faster than it can detect it. Reuters and AP have no RL-based solution for claim verification. Every major platform still relies on human reviewers cross-referencing claims against reputable sources and trusted databases, one story at a time.

**ChronoVeritas** is the first RL environment that models this exact workflow: given a claim and a corpus of timestamped documents with varying reliability, the agent must classify the truth status, identify the mutation type, and locate the exact source where the distortion originated.

Misinformation rarely appears from nothing. It evolves: a government report states a 5% budget increase; a news article rounds it to "nearly 10%"; a blog post claims "budgets doubled." ChronoVeritas models this real-world mutation process and challenges AI agents to:

1. **Search** a corpus of documents with varying reliability tiers
2. **Reconstruct** the chronological timeline of how a claim evolved
3. **Identify** the exact document where the mutation occurred
4. **Classify** the mutation type (distortion, omission, fabrication, context shift)
5. **Deliver** a verdict with calibrated confidence

Unlike simple binary fact-checking benchmarks, ChronoVeritas requires agents to perform **multi-hop reasoning across conflicting sources** — the core skill needed for real-world misinformation forensics.

### Why Reinforcement Learning?

This problem has **genuine non-linear structure** that makes it ideal for RL:

- An agent that masters verdict accuracy still scores poorly if it can't localize the mutation point.
- An agent that retrieves exhaustively still gets penalized on efficiency.
- The agent must decide **when to re-search** (backtracking is expensive), **which contradictions are meaningful** vs. noise, and **how far back to trace** a claim's origin.
- Every decision has a **cost-vs-reward tradeoff** — exactly what RL is built to optimize.

This is exactly the kind of environment that differentiates **RL-trained agents from prompt-engineered ones** — the multi-dimensional reward surface cannot be solved by a single chain-of-thought prompt.

---

## Environment Design

### Observation Space

At each step, the agent observes:

| Field | Type | Description |
|---|---|---|
| `claim` | `str` | The factual claim to investigate |
| `corpus_metadata` | `List[DocMeta]` | Metadata stubs for discovered documents (initially empty) |
| `retrieved_docs` | `List[Document]` | Full text of fetched documents |
| `agent_timeline` | `List[TimelineEntry]` | Agent-built chronological event timeline |
| `flagged_contradictions` | `List[Tuple[str, str]]` | Pairs of contradictory documents flagged by the agent |
| `current_step` | `int` | Current step number |
| `max_steps` | `int` | Maximum steps before budget exhaustion |
| `token_budget_remaining` | `int` | Remaining token budget for fetching documents |
| `partial_reward_so_far` | `float` | Accumulated process rewards |

**Key design choice:** The corpus starts **empty**. Agents must use `search` to discover documents before they can read them. This prevents information leakage from the initial observation.

Each document includes a **reliability tier** visible at search time:

| Tier | Label | Examples | Trust Weight |
|------|-------|----------|-------------|
| 1 | Official | Government filings, court records, peer-reviewed research | 1.0 |
| 2 | Institutional | Major news organizations, corporate press releases | 0.6 |
| 3 | Informal | Blogs, forums, social media, leaked documents | 0.2 |

### Action Space

| Action | Step Cost | Token Cost | Description |
|---|---|---|---|
| `search` | 1 step | 0 | BM25 keyword search — discovers document metadata |
| `fetch_doc` | 1 step | `len(content) // 4` | Retrieves full document text |
| `add_timeline_event` | **0 (free)** | 0 | Annotates a chronological event from a read document |
| `flag_contradiction` | **0 (free)** | 0 | Marks two documents as containing contradictory facts |
| `set_mutation_point` | **0 (free)** | 0 | Declares which document was mutated and how |
| `submit_verdict` | Terminal | 0 | Submits final verdict — triggers grading, ends episode |

**Two resource budgets constrain the agent:**
- **Step budget** — Only `search` and `fetch_doc` consume steps. Free actions (timeline, contradiction, mutation, verdict) allow unlimited annotation without penalty.
- **Token budget** — Fetching documents deducts tokens proportional to content length (`len(content) // 4`). This forces agents to prioritize which documents to read in full.

### Episode Lifecycle

```
reset(task_id)
  → Agent sees: claim text, empty corpus, budget limits
  → Phase: INITIALISED

search(query) / fetch_doc(doc_id)    [consumes steps/tokens]
  → Agent discovers and reads documents
  → Reward: 0.0 (no answer leakage)

add_timeline_event / flag_contradiction    [free actions]
  → Agent annotates evidence graph
  → Reward: small process bonuses (≤0.02 each)

set_mutation_point(doc_id, mutation_type)    [free action]
  → Agent declares its hypothesis
  → Reward: 0.0 (CRITICAL: no ground-truth comparison)

submit_verdict(verdict, mutation_type, provenance_chain, confidence)
  → Terminal grading fires — ONLY point where GT is compared
  → Phase: TERMINAL
```

**Episode termination:** Either `submit_verdict` or step budget exhaustion (partial grading applies).

---

## Reward Design

### Core Principle: Zero Answer Leakage

> *"Could an agent that ignores all document content and only reads reward signals achieve a non-trivial score?"* — If yes, the reward function is broken.

ChronoVeritas implements a **process-oriented reward system** inspired by Process Reward Models (PRMs) in LLM alignment research. Intermediate rewards measure **reasoning quality**, not proximity to the answer.

### Intermediate Rewards (Non-Exploitable)

| Action | Max Reward | What Is Measured |
|---|---|---|
| `add_timeline_event` | +0.02 per event | Did the agent read the document? Does the timestamp match? Is the label grounded in content? |
| `flag_contradiction` | +0.02 per flag | Did the agent actually fetch both documents before flagging? |
| `set_mutation_point` | **0.0 always** | Records declaration for terminal grading. Returns evidence-grounding feedback only. |

**Caps:** Timeline bonuses capped at 0.10 (5 events). Contradiction bonuses capped at 0.04 (2 flags). Maximum total intermediate reward: ~0.14.

### Terminal Grading (10 Scoring Components)

All ground-truth comparison happens **exclusively** at `submit_verdict`. The grading system uses **10 components**, weighted differently per difficulty tier:

| Component | Range | Easy | Medium | Hard | Description |
|---|---|---|---|---|---|
| `verdict_accuracy` | {0, 1} | **0.45** | **0.25** | **0.20** | Exact match: `true`, `false`, `misleading`, `unverifiable` |
| `source_reliability` | [0, 1] | 0.10 | 0.10 | 0.08 | Weighted average reliability tier of provenance chain |
| `mutation_type_score` | {0, 1} | — | **0.20** | **0.10** | Exact match: `distortion`, `omission`, `fabrication`, `context_shift` |
| `mutation_point_score` | {0, 0.5, 1} | **0.35** | **0.20** | **0.10** | 1.0 = exact doc match · 0.5 = adjacent in timeline · 0.0 = wrong |
| `provenance_chain` | [0, 1] | 0.05 | 0.10 | **0.15** | Multiset-aware F1 between agent and ground-truth chains |
| `timeline_score` | [0, 1] | — | — | **0.07** | Kendall-tau correlation of agent's timeline ordering |
| `efficiency_score` | [0, 1] | — | 0.10 | 0.05 | `1 − steps_used / max_steps` — rewards concise investigation |
| `early_detection` | {0, 1} | 0.05 | 0.05 | 0.05 | Correct mutation point identified before 40% of step budget |
| `reconciliation` | [0, 1] | — | — | **0.15** | Hard only: coverage of conflict fields via contradiction flags |
| Σ positive | | **1.00** | **1.00** | **0.95** | Hard tasks ceiling at 0.95 by design |
| `hallucination_penalty` | −[0, 1] | −0.10 | −0.10 | **−0.15** | Penalty for citing unfetched or fabricated documents |
| `brier_penalty` | −[0, 1] | −0.05 | −0.08 | **−0.10** | Penalty for miscalibrated confidence |

**Why hard positives sum to 0.95:** The reconciliation component (0.15) creates an intentional ceiling — hard tasks should never trivially reach 1.0. This forces mastery across ALL scoring dimensions.

### Determinism Guarantee

All grading is **fully deterministic and reproducible** — no LLM-as-judge, no stochastic evaluation. The same agent actions always produce the same score. Every scoring function is a pure mathematical computation over typed Pydantic models.

---

## Tasks

ChronoVeritas ships with 3 tasks spanning a clear difficulty progression:

### EASY-001 — Budget Distortion (15 steps, 4 docs)

> *"The proposed transit budget increases by 15%..."*

A city council approves a 5% budget increase (official minutes). A secondary news source distorts "5%" to "15%." The agent must identify the numerical distortion and trace it to the source.

- **Mutation:** `distortion` — a specific number is altered
- **Corpus:** 4 documents across 3 reliability tiers
- **Key challenge:** Simple numerical comparison between authoritative and secondary sources

### MED-001 — Corporate Restructuring Fabrication (20 steps, 8 docs)

> *"GlobalTech Corp involuntarily terminated 1,200 employees amid a 12% revenue decline..."*

An SEC filing shows 800 employees transferred (not terminated) and 400 accepted voluntary separation. Revenue grew 3.1%, not declined 12%. A news article fabricates both the revenue decline and the characterization of involuntary terminations.

- **Mutation:** `fabrication` — financial data and event characterization are invented
- **Corpus:** 8 documents including SEC filings (T1), news reports (T2), employee forums (T3)
- **Key challenge:** Reconciling corporate PR with official filings, identifying fabricated statistics

### HARD-001 — Aviation Safety Certification Fabrication (30 steps, 20 docs)

> *"The FAA conducted fully independent flight-control safety certification of the Boeing 737 MAX's MCAS system..."*

A dense investigation spanning 5 years of the Boeing 737 MAX crisis. A fabricated "leaked FAA memo" (DOC-5005) claims independent MCAS testing was performed, directly contradicted by 7 official sources including congressional reports, DOT Inspector General findings, and DOJ settlement admissions.

- **Mutation:** `fabrication` — an entire document is forged
- **Corpus:** 20 documents, 9-doc provenance chain, 15-doc timeline, 5 conflict fields, 5 noise documents
- **Key challenges:**
  - 5 conflict fields requiring multi-source reconciliation (certification independence, MCAS test performance, pilot disclosure, ODA delegation scope, sensor redundancy)
  - Noise documents (competitor announcements, engine specs, space program) that distract from the signal
  - Subtle fabrication — the forged memo uses plausible bureaucratic language and references real regulatory frameworks
  - Requires anchoring on Tier-1 sources (FAA reports, congressional testimony) over Tier-3 leaked documents

### Difficulty Progression

| Property | Easy | Medium | Hard |
|---|---|---|---|
| Corpus size | 4 docs | 8 docs | 20 docs |
| Provenance chain | 3 docs | 3 docs | 9 docs |
| Timeline depth | 4 docs | 8 docs | 15 docs |
| Conflict fields | 0 | 0 | 5 |
| Noise documents | 0 | 0 | 5 |
| Max steps | 15 | 20 | 30 |
| Grader components | 5 | 7 | 9 |

### Expected Baseline Scores

| Difficulty | Random Agent | Baseline LLM (8B) | Frontier Model |
|---|---|---|---|
| Easy | 0.10–0.15 | 0.85–0.95 | 0.95+ |
| Medium | 0.05–0.10 | 0.60–0.80 | 0.85–0.95 |
| Hard | 0.02–0.05 | 0.40–0.65 | 0.70–0.85 |

---

## Project Structure

```
chronoveritas/
├── server.py                   # FastAPI server (OpenEnv HTTP API)
├── inference.py                # Baseline inference agent
├── openenv.yaml                # OpenEnv specification manifest
├── Dockerfile                  # Container deployment
├── requirements.txt            # Python dependencies
├── pyproject.toml              # Package metadata
│
├── env/                        # Core environment
│   ├── environment.py          # Episode lifecycle & step dispatch
│   ├── actions.py              # Action handlers (search, fetch, annotate, grade)
│   ├── models.py               # Pydantic v2 typed models (Document, Action, Observation, etc.)
│   └── state_manager.py        # Mutable episode state with phase transitions
│
├── graders/                    # Deterministic grading system
│   ├── base_grader.py          # Abstract grader with 10 shared scoring components
│   ├── easy_grader.py          # EasyGrader (5 components, verdict-heavy)
│   ├── medium_grader.py        # MediumGrader (7 components, +mutation type & efficiency)
│   └── hard_grader.py          # HardGrader (9 components, +reconciliation & timeline)
│
├── search/                     # Retrieval engine
│   └── bm25_index.py           # BM25 corpus indexing and keyword search
│
└── data/
    └── tasks/                  # Task definitions (JSON)
        ├── task_easy.json
        ├── task_medium.json
        └── task_hard.json
```

---

## Quick Start

### Local Development

```bash
# Clone and install
git clone <repo-url>
cd chronoveritas
pip install -r requirements.txt

# Start the environment server
uvicorn server:app --host 0.0.0.0 --port 7860

# Health check
curl http://localhost:7860/health
```

### Running the Baseline Agent

```bash
# Set your LLM API credentials
export API_BASE_URL=https://api.openai.com/v1
export MODEL_NAME=gpt-4o-mini
export OPENAI_API_KEY=sk-...
export ENV_BASE_URL=http://localhost:7860

# Run inference across all 3 tasks
python inference.py
```

### Docker Deployment

```bash
docker build -t chronoveritas .
docker run -p 7860:7860 chronoveritas
```

---

## API Reference

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Returns `{"status": "healthy"}` |
| `GET` | `/tasks` | Lists available task IDs and difficulty levels |
| `POST` | `/reset` | Starts a new episode for a given task |
| `POST` | `/step` | Executes a single agent action |
| `GET` | `/state` | Returns the current observation |

### Example: Full Episode

```bash
# 1. Reset to a task
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "EASY-001"}'

# 2. Search for documents
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"type": "search", "payload": {"query": "budget transport council"}}'

# 3. Fetch a discovered document
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"type": "fetch_doc", "payload": {"doc_id": "DOC-0001"}}'

# 4. Annotate timeline (free action)
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"type": "add_timeline_event", "payload": {"doc_id": "DOC-0001", "event_label": "Council approved 5% increase", "timestamp": 1672531200}}'

# 5. Flag contradiction (free action)
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"type": "flag_contradiction", "payload": {"doc_id_a": "DOC-0001", "doc_id_b": "DOC-0003"}}'

# 6. Declare mutation point (free action, returns 0 reward)
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"type": "set_mutation_point", "payload": {"doc_id": "DOC-0003", "mutation_type": "distortion"}}'

# 7. Submit final verdict (terminal — triggers grading)
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"type": "submit_verdict", "payload": {"verdict": "false", "mutation_type": "distortion", "mutation_doc_id": "DOC-0003", "provenance_chain": ["DOC-0001", "DOC-0003"], "confidence": 0.9}}'
```

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `API_BASE_URL` | `https://api.openai.com/v1` | LLM API endpoint for inference agent |
| `MODEL_NAME` | `gpt-4o-mini` | LLM model name |
| `HF_TOKEN` | *(required)* | API key / Hugging Face token for LLM calls |
| `ENV_BASE_URL` | `https://transyltoonia-chronoveritas.hf.space` | Environment server URL |

---

## Technical Highlights

### Why ChronoVeritas is Different

| Feature | Typical Fact-Check Benchmarks | ChronoVeritas |
|---|---|---|
| Task structure | Binary true/false classification | Multi-step investigation with provenance tracking |
| Evidence model | Flat document retrieval | Hierarchical reliability tiers + temporal ordering |
| Reward design | Final accuracy only | Process rewards + 10-component terminal grading |
| Answer leakage | Often present in intermediate signals | **Zero leakage** — `set_mutation_point` always returns 0.0 |
| Grading | LLM-as-judge or human evaluation | **Fully deterministic** — pure mathematical scoring |
| Difficulty curve | Single difficulty level | 3 tiers with distinct grader architectures |
| RL suitability | Static accuracy benchmarks | Non-linear reward surface with cost–benefit tradeoffs at every step |

### Anti-Exploitation Design

1. **No GT Oracle:** `set_mutation_point` records the declaration silently — reward is always 0.0. A brute-force agent learns nothing from intermediate rewards.
2. **Evidence Grounding:** Process rewards only check if the agent *read its evidence* — not whether the evidence is *correct*.
3. **Hallucination Detection:** Citing documents the agent never fetched triggers penalties.
4. **Brier Penalty:** Overconfident wrong answers and underconfident correct answers are both penalized — encouraging calibrated reasoning.
5. **Intentional Ceiling:** Hard task positive weights sum to 0.95, preventing trivial perfect scores.

### Real-World Impact

ChronoVeritas is built to be **directly useful** — not just an academic exercise:

- **Content moderation at scale:** Platforms like Meta, X, and YouTube spend $200M+/year on human fact-checkers. An RL agent trained on ChronoVeritas-style tasks could triage the ~95% of flagged content that follows known mutation patterns, letting human reviewers focus on novel or ambiguous cases.
- **Newsroom verification:** A Reuters journalist currently spends 30–90 minutes manually cross-referencing a breaking claim against archived sources. An agent that can search, retrieve, compare reliability tiers, and flag contradictions could reduce this to seconds — with a provenance chain the journalist can audit.
- **Financial compliance:** The SEC mandates that public companies' current statements be consistent with prior filings. Today, compliance teams do this manually. ChronoVeritas trains the exact skill needed: detect when a current claim contradicts an authoritative prior source.
- **Combating AI-generated misinformation:** As LLMs like Grok generate plausible-sounding but factually incorrect content at scale, the need for automated claim verification grows exponentially. The mutation types in ChronoVeritas (distortion, omission, fabrication, context shift) directly map to the ways AI hallucinations distort real information.

The environment's reward structure is specifically designed so that RL training produces agents with skills that transfer to production: source prioritization, efficient investigation under budget constraints, confidence calibration, and provenance tracking — not just final-answer accuracy.

---

## License

MIT

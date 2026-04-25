# ChronoVeritas v2 — Complete Build Spec
## "Can an AI reverse-engineer a lie?"
### For Cursor / Claude: Read this entirely before touching any file.

---

## WHAT EXISTS (Keep All of V1 — Do Not Modify)
```
env/environment.py      ✅ keep as-is
env/models.py           ✅ keep as-is  
env/state_manager.py    ✅ keep as-is
graders/base_grader.py  ✅ keep as-is
graders/easy_grader.py  ✅ keep as-is
graders/medium_grader.py ✅ keep as-is
graders/hard_grader.py  ✅ keep as-is
search/bm25_index.py    ✅ keep as-is
search/corpus_store.py  ✅ keep as-is
data/tasks/*.json       ✅ keep as-is
```

## WHAT TO BUILD (New in V2)
```
agents/mutator.py       🆕 Rule-based claim mutator
agents/spreader.py      🆕 Rule-based corpus/task builder
agents/task_bank.py     🆕 Static seed facts for task generation
train_grpo.py           🆕 GRPO training with Unsloth (MOST CRITICAL)
eval.py                 🆕 Evaluation + reward plot generation
server.py               🔄 Update to expose /mutate and /spread endpoints
openenv.yaml            🆕 OpenEnv manifest
requirements.txt        🆕 Full deps
Dockerfile              🆕 HF Spaces deployment
README.md               🆕 Story + plots
```

---

## THE STORY (Judges Must Understand This in 30 Seconds)

> "Misinformation has three actors: someone who distorts a fact (Mutator),
>  someone who spreads it through documents (Spreader), and someone who
>  must detect it (Fact-Checker). We train the Fact-Checker with RL."

**Demo flow:**
1. Mutator takes a true claim → distorts it (e.g., "5% budget rise" → "15% budget rise")
2. Spreader creates a corpus of 3–8 documents embedding the distorted claim
3. Fact-Checker (Qwen2.5-7B-Instruct, GRPO-trained) investigates and identifies the mutation
4. Reward goes up over training → show the curve

---

## ARCHITECTURE

### Training Mode (GRPO — Single Turn)
```
Prompt:
  [SYSTEM] You are a misinformation forensics expert...
  [USER]   CLAIM: "The transit budget increases by 15%"
           DOCUMENTS:
           [DOC-001] Official minutes — budget approved at 5%
           [DOC-002] News article — "nearly 15% increase approved"
           [DOC-003] Blog post — "council doubles transit budget"
  
Model Output (JSON):
  {
    "verdict": "false",
    "mutation_type": "distortion", 
    "mutation_doc_id": "DOC-002",
    "provenance_chain": ["DOC-001", "DOC-002"],
    "confidence": 0.92
  }

Reward: computed by existing deterministic graders (same weights as eval)
```

### Evaluation Mode (Multi-Step OpenEnv)
- Uses the existing v1 environment with search/fetch/annotate/submit
- Same graders, same tasks
- Shows the full interactive workflow for the demo

---

## MUTATOR DESIGN (agents/mutator.py)

The Mutator is a **rule-based script** (not an LLM). It takes a Document with a true claim and applies one of 4 mutations:

### Mutation Rules

**DISTORTION** — alter a number/name/date
- Find numbers via regex: `r'\b\d+(?:\.\d+)?%?\b'`
- Apply multiplier: randomly pick 1.5x, 2x, 3x, or 0.3x
- Example: "5% increase" → "15% increase" (3x), "approved 800 employees" → "1,200 employees" (1.5x)
- Also handles names: pick from a list of plausible alternates

**FABRICATION** — invent a statistic or event
- Templates: `"{entity} announced {fabricated_number}% {direction} in {metric}"`
- The invented number must differ from any real number in the document by >50%
- Example: "Revenue grew 3.1%" → "Revenue declined 12%"

**OMISSION** — remove a key qualifier
- Find: "voluntary", "transferred", "pending approval", "allegedly", "estimated"
- Remove or replace with its opposite
- Example: "400 accepted voluntary separation" → "400 were terminated"

**CONTEXT_SHIFT** — true fact, wrong frame
- Find a real data point and attribute it to wrong period/entity/scope
- Example: "Q3 profit rose 8%" → "Annual profit rose 8%" (wrong scope)

### Mutator Output
```python
@dataclass
class MutationResult:
    original_doc_id: str
    mutated_content: str      # The altered document text
    mutation_type: str        # which of the 4 types
    true_claim: str           # what was originally true
    false_claim: str          # the claim the agent must investigate
    diff_description: str     # human-readable explanation for demo
```

---

## SPREADER DESIGN (agents/spreader.py)

The Spreader takes a MutationResult and builds a full TaskSpec (compatible with the v1 TaskSpec model).

### Document Corpus Structure

**Easy (3 docs, 0 noise):**
```
DOC-001  Tier 1  Original true source (e.g., official minutes)
DOC-002  Tier 2  Mutation point — first distorted version (news article)
DOC-003  Tier 3  Amplified version (blog post, references DOC-002's distortion)
```

**Medium (6 docs, 1 noise):**
```
DOC-001  Tier 1  Primary source (true)
DOC-002  Tier 1  Corroborating source (true)  
DOC-003  Tier 2  Mutation point (distorted)
DOC-004  Tier 2  Secondary spread (references DOC-003)
DOC-005  Tier 3  Tertiary spread (amplified)
DOC-006  Tier 3  NOISE — unrelated story, same domain
```

**Hard (12 docs, 3 noise):**
```
DOC-001 to DOC-004  Tier 1  Multiple authoritative true sources
DOC-005             Tier 2  Mutation point
DOC-006 to DOC-009  Tier 2/3  Spread chain
DOC-010 to DOC-012  Any tier  NOISE documents (distractors)
```

### Document Content Templates
The Spreader uses fill-in-the-blank templates for each tier and domain.
Templates are in `agents/task_bank.py`.

### Spreader Output
A complete `TaskSpec` dict that can be:
1. Written to `data/tasks/generated_*.json` for persistent training tasks
2. Used directly in GRPO training without writing to disk

---

## REWARD FUNCTION (train_grpo.py — most critical section)

### Five Independent Reward Components
These map 1:1 to the existing grader components but are computed in a 
single-turn format for GRPO:

```python
def compute_reward(completion: str, ground_truth: dict) -> tuple[float, dict]:
    """
    Returns (total_reward, breakdown_dict) for logging.
    All components independent — cannot be hacked individually.
    """
    parsed = extract_json_safe(completion)
    
    # ── GATE: format penalty ──────────────────────────────────────
    if parsed is None:
        return -0.15, {"format": -0.15}  # Hard penalty for invalid JSON
    
    required = {"verdict", "mutation_type", "mutation_doc_id", "confidence"}
    if not required.issubset(parsed.keys()):
        return -0.10, {"format": -0.10}  # Missing fields
    
    breakdown = {}
    
    # ── Component 1: Format valid (+0.05) ───────────────────────
    breakdown["format"] = 0.05
    
    # ── Component 2: Verdict accuracy (+0.35) ────────────────────
    verdict_score = 1.0 if parsed["verdict"] == ground_truth["gt_verdict"] else 0.0
    breakdown["verdict"] = 0.35 * verdict_score
    
    # ── Component 3: Mutation type (+0.25) ───────────────────────
    mut_type_score = 1.0 if parsed["mutation_type"] == ground_truth["gt_mutation_type"] else 0.0
    breakdown["mutation_type"] = 0.25 * mut_type_score
    
    # ── Component 4: Mutation point (+0.25) ──────────────────────
    pred_doc = str(parsed.get("mutation_doc_id", "")).strip()
    true_doc = ground_truth["gt_mutation_doc_id"]
    gt_timeline = ground_truth.get("gt_timeline", [])
    
    if pred_doc == true_doc:
        mp_score = 1.0
    elif (pred_doc in gt_timeline and true_doc in gt_timeline and
          abs(gt_timeline.index(pred_doc) - gt_timeline.index(true_doc)) == 1):
        mp_score = 0.5   # Adjacent in timeline → partial credit
    else:
        mp_score = 0.0
    breakdown["mutation_point"] = 0.25 * mp_score
    
    # ── Component 5: Calibration (+0.05) ─────────────────────────
    conf = float(parsed.get("confidence", 0.5))
    conf = max(0.0, min(1.0, conf))
    correct = verdict_score == 1.0
    brier = (conf - (1.0 if correct else 0.0)) ** 2
    calibration_score = max(0.0, 1.0 - brier)
    breakdown["calibration"] = 0.05 * calibration_score
    
    # ── Hallucination penalty (−0.05 per fabricated doc) ─────────
    corpus_ids = set(ground_truth.get("corpus_ids", []))
    prov = parsed.get("provenance_chain", [])
    if isinstance(prov, list):
        fabricated = [d for d in prov if d not in corpus_ids]
        hallucination_penalty = min(0.20, len(fabricated) * 0.05)
        breakdown["hallucination_penalty"] = -hallucination_penalty
    else:
        breakdown["hallucination_penalty"] = 0.0
    
    total = sum(breakdown.values())
    return max(-0.20, min(1.0, total)), breakdown
```

### Why This Cannot Be Gamed
1. **Format gate:** No reward for invalid JSON — model can't skip to verdict
2. **Independent components:** Gaming verdict alone gets 0.35 max, not 1.0
3. **Partial credit traps:** Adjacent doc gets 0.5 not 0.0 — prevents cliff effects that cause training instability
4. **Calibration component:** Overconfident wrong answers and underconfident right answers both penalised
5. **Hallucination floor:** −0.20 floor prevents negative spiral while still penalising fabrication

---

## GRPO TRAINING LOOP (train_grpo.py)

### GPU / Memory Config (RTX A4500, 20GB VRAM)
```python
model_name = "unsloth/Qwen2.5-7B-Instruct"
max_seq_length = 2048     # enough for easy+medium; trim hard tasks
load_in_4bit = True       # ~5GB model, leaves 15GB for training
lora_r = 16
lora_alpha = 16
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"]

# GRPO config
num_generations = 8       # group size — 8 rollouts per prompt
per_device_train_batch_size = 1
gradient_accumulation_steps = 4   # effective batch = 4
learning_rate = 5e-5
max_grad_norm = 0.1
```

### Curriculum Schedule (Total ~400 steps, ~3-4 hours on A4500)
```
Steps   0–150   Easy tasks only    (high initial success rate → establishes format)
Steps 150–300   Easy + Medium mix  (adds mutation_type learning)
Steps 300–400   All difficulties   (pushes provenance chain + hard reasoning)
```

### Dataset Construction
```python
def build_grpo_dataset(tasks: list[TaskSpec], repeats: int = 8) -> Dataset:
    """
    Each task becomes `repeats` identical prompt rows.
    GRPO samples all 8 in a group, computes advantages, updates model.
    Ground truth stored as metadata — never in the prompt.
    """
    rows = []
    for task in tasks:
        prompt = format_single_turn_prompt(task)
        gt = task.ground_truth.model_dump()
        gt["corpus_ids"] = [d.doc_id for d in task.corpus]
        for _ in range(repeats):
            rows.append({"prompt": prompt, "ground_truth": gt})
    return Dataset.from_list(rows)
```

### Format for Qwen2.5-7B-Instruct (ChatML)
```python
SYSTEM_PROMPT = """You are a misinformation forensics expert. 
Your task: given a CLAIM and a set of DOCUMENTS, determine:
1. Whether the claim is true, false, or misleading
2. What type of mutation occurred (distortion/fabrication/omission/context_shift/none)
3. Which document FIRST introduced the distortion
4. Your confidence level

Always respond with ONLY valid JSON. No explanation before or after the JSON.
Required format:
{"verdict": "true|false|misleading|unverifiable", "mutation_type": "distortion|fabrication|omission|context_shift|none", "mutation_doc_id": "DOC-XXX or null", "provenance_chain": ["DOC-XXX", ...], "confidence": 0.0}"""

def format_single_turn_prompt(task: TaskSpec) -> str:
    tier_map = {1: "Official (Tier 1)", 2: "Institutional (Tier 2)", 3: "Informal (Tier 3)"}
    docs_text = ""
    for doc in task.corpus:
        docs_text += f"\n[{doc.doc_id}] {tier_map[doc.reliability_tier]} | {doc.source} | {doc.date_str if hasattr(doc, 'date_str') else ''}\n"
        docs_text += f"Title: {doc.title}\n"
        docs_text += doc.content + "\n"
        docs_text += "---\n"
    
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n"
        f"CLAIM TO INVESTIGATE: \"{task.claim}\"\n\n"
        f"DOCUMENTS:\n{docs_text}\n"
        f"Analyze the documents and identify the mutation. Respond with JSON only.<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
```

---

## EVAL SCRIPT (eval.py)

Run before and after training. Outputs:
1. `plots/reward_curve.png` — training reward over steps
2. `plots/component_breakdown.png` — per-component reward over steps  
3. `plots/before_after.png` — baseline vs trained model on same tasks
4. `eval_results.json` — all raw numbers for README

```python
# Evaluation pipeline
def evaluate_model(model, tokenizer, tasks, n_samples=3):
    """Run model on tasks, return average reward breakdown."""
    results = []
    for task in tasks:
        prompt = format_single_turn_prompt(task)
        for _ in range(n_samples):
            completion = generate(model, tokenizer, prompt)
            reward, breakdown = compute_reward(completion, task.ground_truth.model_dump())
            results.append({"task_id": task.task_id, "reward": reward, **breakdown})
    return pd.DataFrame(results)
```

---

## SERVER UPDATES (server.py)

Add two new endpoints to expose the multi-agent story:

```
POST /mutate   — Takes a doc_id + mutation_type, returns mutated content
POST /spread   — Takes a mutation result, returns a full TaskSpec
GET  /demo     — Returns a real-time demo of all 3 agents in sequence
```

These endpoints power the HF Spaces demo UI.

---

## OPENENV YAML

```yaml
name: chronoveritas-v2
version: "2.0.0"
description: >
  RL environment for misinformation forensics. Three-agent system:
  Mutator generates distorted claims, Spreader embeds them in a document
  corpus, Fact-Checker (GRPO-trained LLM) identifies the mutation.
author: your-team-name
theme: multi-agent-interactions

actions:
  - name: search
    description: BM25 keyword search over document corpus
  - name: fetch_doc
    description: Retrieve full document content
  - name: add_timeline_event
    description: Annotate chronological event (free action)
  - name: flag_contradiction
    description: Mark two documents as contradictory (free action)
  - name: set_mutation_point
    description: Declare hypothesis — no GT comparison (free action)
  - name: submit_verdict
    description: Terminal action — triggers deterministic grading

difficulty_levels:
  - easy:   { max_steps: 15, corpus_size: 3 }
  - medium: { max_steps: 20, corpus_size: 6 }
  - hard:   { max_steps: 30, corpus_size: 12 }

reward:
  type: deterministic
  components: [verdict, mutation_type, mutation_point, provenance, 
               source_reliability, efficiency, calibration]
  penalties: [hallucination, brier]
  leakage: none

training:
  algorithm: GRPO
  base_model: Qwen/Qwen2.5-7B-Instruct
  framework: unsloth + trl
```

---

## DOCKERFILE

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Create data directories
RUN mkdir -p data/tasks

EXPOSE 7860

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860"]
```

---

## REQUIREMENTS.TXT

```
# Environment
fastapi==0.115.0
uvicorn==0.30.6
pydantic>=2.0.0
rank-bm25==0.2.2

# Training
torch>=2.1.0
unsloth[colab-new]  
trl>=0.12.0
transformers>=4.45.0
datasets>=2.20.0
accelerate>=0.34.0

# Eval & plotting
pandas>=2.0.0
matplotlib>=3.8.0
scipy>=1.11.0
numpy>=1.26.0

# Utils
python-dotenv==1.0.0
httpx==0.27.0
```

---

## EXECUTION ORDER FOR YOUR 24 HOURS

### Hour 1-2: Person A (Environment)
1. Copy all v1 files into v2 repo
2. Write `agents/task_bank.py` (seed facts — 20 true claims across 3 domains)
3. Write `agents/mutator.py` (4 mutation rules with regex)
4. Write `agents/spreader.py` (corpus builder using templates)
5. Run: `python -c "from agents.spreader import Spreader; print(Spreader().generate('easy'))"` — confirm TaskSpec output

### Hour 2-4: Person B (Rewards + Server)  
1. Write `train_grpo.py` reward function (copy from spec above, test with mock data)
2. Update `server.py` with `/mutate` and `/spread` endpoints
3. Write `openenv.yaml`
4. Write `Dockerfile` + `requirements.txt`
5. Test full episode: `uvicorn server:app --port 7860` → curl /reset → curl /step

### Hour 4-8: Person C (Training)
1. Install Unsloth: `pip install unsloth`
2. Run `python train_grpo.py --difficulty easy --steps 50` — verify non-zero reward
3. Inspect generations: are they valid JSON? Are rewards going up?
4. If zero reward: check JSON extraction regex, check prompt format
5. Run full curriculum: `python train_grpo.py --curriculum --steps 400`
6. Save model: `model.save_pretrained("chronoveritas-fact-checker")`

### Hour 8-10: All (Eval + Demo)
1. Run `python eval.py --model chronoveritas-fact-checker --baseline`
2. Generate plots: `plots/reward_curve.png`, `plots/before_after.png`
3. Write `README.md` with embedded plots
4. Deploy to HF Spaces

### Hour 10+: Person D (Story + Polish)
1. Record 90-second demo video: show Mutator → Spreader → Fact-Checker flow
2. Write HF blog post (500 words, link to paper references)
3. Final README check: does it answer Problem/Environment/Results/Why?

---

## ANTI-HACKING CHECKLIST
- [ ] `set_mutation_point` always returns 0.0 reward (verified in base_grader.py ✅)
- [ ] `extract_json_safe` uses try/catch, never crashes on adversarial output
- [ ] Reward floor at -0.20 prevents death spiral
- [ ] Hallucination check uses exact doc_id string match, no fuzzy
- [ ] Confidence is clamped to [0.0, 1.0] before brier computation
- [ ] No GT data in the prompt (ground_truth dict stored separately)
- [ ] Generated tasks saved to separate dir from eval tasks (no train/test leakage)

---

## PLOT SPEC (what judges need to see)

### Plot 1: Training Reward Curve (reward_curve.png)
- X-axis: Training step (0–400)
- Y-axis: Average episode reward (0.0–1.0)
- Lines: "Easy tasks", "Medium tasks" (rolling average window=20)
- Expected shape: starts ~0.1, rises to ~0.6–0.8 on easy, ~0.4–0.6 on medium
- Save as PNG at 150 DPI

### Plot 2: Component Breakdown (component_breakdown.png)
- X-axis: Training step
- Y-axis: Reward component value (0.0–0.35)
- Stacked area chart: verdict / mutation_type / mutation_point / calibration
- Shows the model learning each skill in order

### Plot 3: Before vs After (before_after.png)
- Bar chart: 5 tasks × 2 bars (baseline / trained)
- Y-axis: Total reward
- Use consistent colors (blue=baseline, orange=trained)
- Difference arrow annotation: "+0.42" etc.

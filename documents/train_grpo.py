"""
ChronoVeritas v2 — GRPO Training Script
Uses Unsloth + TRL GRPOTrainer to train Qwen2.5-7B-Instruct as a Fact-Checker.

GPU target: RTX A4500 (20GB VRAM)
Expected runtime: ~3-4 hours for full curriculum (400 steps)

Usage:
  python train_grpo.py                          # Full curriculum
  python train_grpo.py --difficulty easy        # Easy only (debug)
  python train_grpo.py --steps 50 --difficulty easy  # Quick sanity check
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ── Argument parsing (before heavy imports) ────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--difficulty", choices=["easy", "medium", "hard", "curriculum"], default="curriculum")
parser.add_argument("--steps", type=int, default=400)
parser.add_argument("--model", default="unsloth/Qwen2.5-7B-Instruct")
parser.add_argument("--output-dir", default="./chronoveritas-fact-checker")
parser.add_argument("--log-dir", default="./training_logs")
parser.add_argument("--group-size", type=int, default=8)
parser.add_argument("--lr", type=float, default=5e-5)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--generate-tasks", action="store_true", help="Generate new tasks before training")
args = parser.parse_args()

# ── Imports ────────────────────────────────────────────────────────────────
import numpy as np
import torch

# Unsloth must be imported before transformers
try:
    from unsloth import FastLanguageModel, is_bfloat16_supported
    UNSLOTH_AVAILABLE = True
except ImportError:
    print("WARNING: Unsloth not installed. Falling back to standard transformers.")
    UNSLOTH_AVAILABLE = False

from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer
from transformers import AutoTokenizer

# Local imports
sys.path.insert(0, str(Path(__file__).parent))
from agents.task_bank import SEED_FACTS, get_random_fact
from agents.mutator import Mutator
from agents.spreader import Spreader

# ── Constants ──────────────────────────────────────────────────────────────
MAX_SEQ_LEN = 2048
TASKS_DIR = Path("data/tasks")
LOG_DIR = Path(args.log_dir)
LOG_DIR.mkdir(parents=True, exist_ok=True)

SYSTEM_PROMPT = """You are a misinformation forensics expert. Your task is to investigate a CLAIM by analysing a set of DOCUMENTS with different reliability tiers.

RELIABILITY TIERS:
- Tier 1 (Official): Government filings, court records, peer-reviewed research → highest trust
- Tier 2 (Institutional): News organisations, press releases → medium trust
- Tier 3 (Informal): Blogs, forums, social media → lowest trust

YOUR TASK:
1. Read all documents chronologically
2. Identify whether the claim is true, false, or misleading
3. Determine what type of mutation occurred (if any)
4. Find the FIRST document that introduced the distortion
5. List the chain of documents that propagated the false claim

MUTATION TYPES:
- distortion: A real number, name, or date was altered
- fabrication: A statistic or event was invented entirely
- omission: A key qualifying word was removed, changing the meaning
- context_shift: A true fact was placed in the wrong temporal or organisational context
- none: The claim is accurate

RESPOND WITH ONLY VALID JSON — no text before or after:
{"verdict": "true|false|misleading|unverifiable", "mutation_type": "distortion|fabrication|omission|context_shift|none", "mutation_doc_id": "DOC-XXXX or null", "provenance_chain": ["DOC-XXXX", ...], "confidence": 0.0}"""


# ── Reward computation ─────────────────────────────────────────────────────

def extract_json_safe(text: str) -> Optional[Dict]:
    """
    Robustly extract JSON from model output.
    Tries multiple patterns in order of specificity.
    """
    # Pattern 1: Full JSON object with required fields
    patterns = [
        r'\{[^{}]*"verdict"[^{}]*"mutation_type"[^{}]*"mutation_doc_id"[^{}]*\}',
        r'\{[^{}]*"verdict"[^{}]*\}',
        r'\{.*?\}',
    ]
    for pattern in patterns:
        for m in re.finditer(pattern, text, re.DOTALL):
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                continue
    return None


def compute_reward(
    completion: str,
    ground_truth: Dict[str, Any],
) -> Tuple[float, Dict[str, float]]:
    """
    Multi-component reward function — 5 independent components + 1 penalty.
    
    Returns (total_reward, breakdown_dict).
    
    Anti-gaming properties:
    - Format gate: no reward for invalid JSON
    - Independent components: no single component dominates
    - Calibration: penalises both over and underconfidence
    - Hallucination: penalises citing non-existent docs
    - Floor: -0.20 prevents gradient death spiral
    """
    breakdown: Dict[str, float] = {}

    # ── GATE: JSON validity ────────────────────────────────────────────
    parsed = extract_json_safe(completion)
    if parsed is None:
        return -0.15, {"format": -0.15, "parse_error": 1.0}

    required_fields = {"verdict", "mutation_type", "mutation_doc_id", "confidence"}
    if not required_fields.issubset(parsed.keys()):
        return -0.10, {"format": -0.10, "missing_fields": 1.0}

    # ── Component 1: Format valid (+0.05) ──────────────────────────────
    breakdown["format"] = 0.05

    # ── Component 2: Verdict accuracy (+0.35) ──────────────────────────
    gt_verdict = ground_truth.get("gt_verdict", "")
    verdict_correct = (str(parsed.get("verdict", "")).strip() == gt_verdict)
    breakdown["verdict"] = 0.35 if verdict_correct else 0.0

    # ── Component 3: Mutation type (+0.25) ─────────────────────────────
    gt_mut_type = ground_truth.get("gt_mutation_type", "")
    mut_type_correct = (str(parsed.get("mutation_type", "")).strip() == gt_mut_type)
    breakdown["mutation_type"] = 0.25 if mut_type_correct else 0.0

    # ── Component 4: Mutation point (+0.25) ────────────────────────────
    pred_doc = str(parsed.get("mutation_doc_id", "") or "").strip()
    gt_doc = str(ground_truth.get("gt_mutation_doc_id", "") or "").strip()
    gt_timeline = ground_truth.get("gt_timeline", [])

    if pred_doc == gt_doc:
        mp_score = 0.25
    elif (pred_doc and gt_doc and pred_doc in gt_timeline and gt_doc in gt_timeline
          and abs(gt_timeline.index(pred_doc) - gt_timeline.index(gt_doc)) == 1):
        mp_score = 0.12   # Adjacent in timeline → partial credit
    else:
        mp_score = 0.0
    breakdown["mutation_point"] = mp_score

    # ── Component 5: Calibration (+0.05) ───────────────────────────────
    try:
        conf = float(parsed.get("confidence", 0.5))
        conf = max(0.0, min(1.0, conf))
    except (TypeError, ValueError):
        conf = 0.5
    brier = (conf - (1.0 if verdict_correct else 0.0)) ** 2
    breakdown["calibration"] = round(0.05 * max(0.0, 1.0 - brier), 4)

    # ── Hallucination penalty (−0.05 per fabricated doc) ───────────────
    corpus_ids = set(ground_truth.get("corpus_ids", []))
    prov = parsed.get("provenance_chain")
    if isinstance(prov, list):
        fabricated = [d for d in prov if str(d) not in corpus_ids]
        hallucination_penalty = min(0.20, len(fabricated) * 0.05)
    else:
        hallucination_penalty = 0.0
    breakdown["hallucination_penalty"] = -hallucination_penalty

    total = sum(breakdown.values())
    total = max(-0.20, min(1.0, total))
    return total, breakdown


# ── Prompt formatting ──────────────────────────────────────────────────────

def format_single_turn_prompt(task_dict: Dict) -> str:
    """
    Format a task as a single ChatML-style turn for Qwen2.5-7B-Instruct.
    Documents are presented in chronological order (earliest first).
    """
    tier_labels = {1: "Official (Tier 1)", 2: "Institutional (Tier 2)", 3: "Informal (Tier 3)"}
    docs = task_dict.get("corpus", [])

    # Sort by timestamp ascending
    docs_sorted = sorted(docs, key=lambda d: d.get("timestamp", 0))

    docs_text = ""
    for doc in docs_sorted:
        tier = doc.get("reliability_tier", 2)
        docs_text += (
            f"\n[{doc['doc_id']}] {tier_labels.get(tier, 'Unknown')} "
            f"| Source: {doc.get('source', 'Unknown')} "
            f"| Date: {doc.get('date_str', 'Unknown')}\n"
            f"Title: {doc.get('title', 'Untitled')}\n"
            f"{doc.get('content', '')}\n"
            f"{'─' * 60}\n"
        )

    claim = task_dict.get("claim", "")
    n_docs = len(docs)

    prompt = (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n"
        f"CLAIM TO INVESTIGATE: \"{claim}\"\n\n"
        f"CORPUS ({n_docs} documents, sorted chronologically):\n"
        f"{docs_text}\n"
        f"Investigate the documents and identify the mutation. "
        f"Remember: the mutation point is the FIRST document where the distortion appears.\n"
        f"Respond with JSON only.<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    return prompt


# ── Task loading ───────────────────────────────────────────────────────────

def load_static_tasks(difficulty: Optional[str] = None) -> List[Dict]:
    """Load task JSON files from disk (v1 tasks + any generated tasks)."""
    tasks = []
    pattern = "*.json"
    for path in sorted(TASKS_DIR.glob(pattern)):
        try:
            with open(path) as f:
                t = json.load(f)
            if difficulty is None or t.get("difficulty") == difficulty:
                tasks.append(t)
        except Exception as e:
            print(f"  Warning: skipping {path.name}: {e}")
    return tasks


def generate_dynamic_tasks(
    n_easy: int = 20,
    n_medium: int = 10,
    n_hard: int = 5,
    seed: int = 42,
) -> List[Dict]:
    """
    Use Mutator + Spreader to generate fresh tasks for training.
    Generated tasks are saved to data/tasks/generated/ for reproducibility.
    """
    print(f"\n[TaskGen] Generating {n_easy+n_medium+n_hard} tasks via Mutator+Spreader...")
    TASKS_DIR.mkdir(parents=True, exist_ok=True)
    gen_dir = TASKS_DIR / "generated"
    gen_dir.mkdir(parents=True, exist_ok=True)

    mutator = Mutator(seed=seed)
    spreader = Spreader(seed=seed)
    tasks = []
    rng = __import__("random").Random(seed)
    mutation_types = ["distortion", "fabrication", "omission", "context_shift"]

    for i, (difficulty, n) in enumerate([("easy", n_easy), ("medium", n_medium), ("hard", n_hard)]):
        for j in range(n):
            fact = rng.choice(list(SEED_FACTS))
            mut_type = mutation_types[(i + j) % len(mutation_types)]
            try:
                mutation = mutator.mutate(fact, mutation_type=mut_type)
                task = spreader.spread(mutation, difficulty=difficulty)

                # Add corpus_ids to ground_truth for reward function
                task["ground_truth"]["corpus_ids"] = [d["doc_id"] for d in task["corpus"]]

                # Save to disk
                fname = gen_dir / f"{task['task_id']}.json"
                with open(fname, "w") as f:
                    json.dump(task, f, indent=2)

                tasks.append(task)
                print(f"  [{difficulty}] {task['task_id']} — {mut_type} on {fact.fact_id}")
            except Exception as e:
                print(f"  Warning: failed to generate task {j}: {e}")

    print(f"[TaskGen] Generated {len(tasks)} tasks.")
    return tasks


def build_training_dataset(
    tasks: List[Dict],
    n_repeats: int = 8,
) -> Dataset:
    """
    Build GRPO training dataset.
    Each task is repeated n_repeats times (GRPO samples a group per prompt).
    Ground truth is stored as a JSON string in metadata (never in the prompt).
    """
    rows = []
    for task in tasks:
        prompt = format_single_turn_prompt(task)

        # Enrich ground truth with corpus_ids for hallucination detection
        gt = dict(task.get("ground_truth", {}))
        gt["corpus_ids"] = [d["doc_id"] for d in task.get("corpus", [])]

        for _ in range(n_repeats):
            rows.append({
                "prompt": prompt,
                "ground_truth_json": json.dumps(gt),  # stored as string (HF Dataset limitation)
            })

    dataset = Dataset.from_list(rows)
    print(f"[Dataset] {len(dataset)} rows ({len(tasks)} tasks × {n_repeats} repeats)")
    return dataset


# ── Model loading ──────────────────────────────────────────────────────────

def load_model_and_tokenizer(model_name: str):
    """Load model with Unsloth (or fallback to standard transformers)."""
    print(f"\n[Model] Loading {model_name}...")

    if UNSLOTH_AVAILABLE:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=MAX_SEQ_LEN,
            dtype=None,          # Auto-detect (bfloat16 on A4500)
            load_in_4bit=True,   # ~5GB VRAM
        )
        model = FastLanguageModel.get_peft_model(
            model,
            r=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            lora_alpha=16,
            lora_dropout=0,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=args.seed,
            max_seq_length=MAX_SEQ_LEN,
        )
        print("[Model] Loaded with Unsloth 4-bit + LoRA.")
    else:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print("[Model] Loaded without Unsloth (training may be slower).")

    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


# ── Reward function wrapper for TRL ───────────────────────────────────────

# Logging state (shared across reward calls)
_reward_log: List[Dict] = []
_step_counter = [0]

def reward_fn_wrapper(
    prompts: List[str],
    completions: List[str],
    ground_truth_json: Optional[List[str]] = None,
    **kwargs,
) -> List[float]:
    """
    TRL GRPOTrainer reward function signature.
    Logs per-component rewards for plotting.
    """
    rewards = []
    _step_counter[0] += 1
    step = _step_counter[0]

    for i, (completion, gt_json) in enumerate(
        zip(completions, ground_truth_json or ["{}"] * len(completions))
    ):
        try:
            gt = json.loads(gt_json)
        except Exception:
            gt = {}

        total, breakdown = compute_reward(completion, gt)
        rewards.append(total)

        # Log every group[0] (first completion per task)
        if i % args.group_size == 0:
            log_entry = {
                "step": step,
                "total_reward": total,
                **breakdown,
                "completion_preview": completion[:80].replace("\n", " "),
            }
            _reward_log.append(log_entry)

            # Console progress
            if step % 10 == 0 and i == 0:
                verdict = extract_json_safe(completion)
                verdict_str = verdict.get("verdict", "?") if verdict else "INVALID_JSON"
                print(
                    f"  [Step {step:4d}] reward={total:.3f} | "
                    f"verdict={verdict:.4s} | "
                    f"breakdown={breakdown}"
                )

    # Save logs every 25 steps
    if step % 25 == 0:
        _flush_logs(step)

    return rewards


def _flush_logs(step: int) -> None:
    """Save reward logs to disk for eval.py to plot."""
    log_path = LOG_DIR / "reward_log.json"
    with open(log_path, "w") as f:
        json.dump(_reward_log, f, indent=2)

    # Also save a simple CSV for quick inspection
    csv_path = LOG_DIR / "reward_log.csv"
    if _reward_log:
        keys = list(_reward_log[0].keys())
        with open(csv_path, "w") as f:
            f.write(",".join(keys) + "\n")
            for row in _reward_log:
                f.write(",".join(str(row.get(k, "")) for k in keys) + "\n")


# ── Curriculum manager ─────────────────────────────────────────────────────

class CurriculumManager:
    """
    Manages the three-phase curriculum:
      Phase 1 (steps 0–37%):   Easy only         → learn format + basic verdict
      Phase 2 (steps 37–75%):  Easy + Medium mix  → learn mutation_type
      Phase 3 (steps 75–100%): All difficulties   → learn provenance + hard reasoning
    """

    def __init__(self, total_steps: int, all_tasks: Dict[str, List[Dict]]) -> None:
        self.total_steps = total_steps
        self.all_tasks = all_tasks  # {"easy": [...], "medium": [...], "hard": [...]}
        self.phase_boundaries = [0.37, 0.75, 1.0]

    def get_tasks_for_step(self, step: int) -> List[Dict]:
        pct = step / max(self.total_steps, 1)
        if pct < self.phase_boundaries[0]:
            return self.all_tasks.get("easy", [])
        elif pct < self.phase_boundaries[1]:
            easy = self.all_tasks.get("easy", [])
            medium = self.all_tasks.get("medium", [])
            # 70% easy, 30% medium in phase 2
            n = max(len(easy), 1)
            n_med = max(len(medium), 1)
            return easy + medium[:max(1, n_med // 3)]
        else:
            return (
                self.all_tasks.get("easy", [])
                + self.all_tasks.get("medium", [])
                + self.all_tasks.get("hard", [])
            )

    def get_phase_name(self, step: int) -> str:
        pct = step / max(self.total_steps, 1)
        if pct < self.phase_boundaries[0]:
            return "Phase 1 — Easy (format + verdict)"
        elif pct < self.phase_boundaries[1]:
            return "Phase 2 — Easy+Medium (mutation_type)"
        else:
            return "Phase 3 — All (provenance + hard)"


# ── Main training loop ─────────────────────────────────────────────────────

def main() -> None:
    print("=" * 70)
    print("ChronoVeritas v2 — GRPO Training")
    print(f"  Model:      {args.model}")
    print(f"  Difficulty: {args.difficulty}")
    print(f"  Steps:      {args.steps}")
    print(f"  Group size: {args.group_size}")
    print(f"  Output:     {args.output_dir}")
    print("=" * 70)

    # ── Step 1: Load / generate tasks ─────────────────────────────────
    print("\n[1/5] Loading tasks...")
    if args.generate_tasks:
        tasks_easy   = generate_dynamic_tasks(n_easy=20, n_medium=0, n_hard=0)
        tasks_medium = generate_dynamic_tasks(n_easy=0, n_medium=10, n_hard=0, seed=43)
        tasks_hard   = generate_dynamic_tasks(n_easy=0, n_medium=0, n_hard=5, seed=44)
    else:
        # Load from disk (v1 tasks + any previously generated tasks)
        tasks_easy   = load_static_tasks("easy")
        tasks_medium = load_static_tasks("medium")
        tasks_hard   = load_static_tasks("hard")
        # Always generate at least some dynamic tasks
        if len(tasks_easy) < 3:
            print("  Fewer than 3 easy tasks on disk — generating via Mutator+Spreader...")
            tasks_easy += generate_dynamic_tasks(n_easy=15, n_medium=0, n_hard=0)
        if len(tasks_medium) < 2:
            tasks_medium += generate_dynamic_tasks(n_easy=0, n_medium=8, n_hard=0, seed=43)

    print(f"  Tasks: easy={len(tasks_easy)}, medium={len(tasks_medium)}, hard={len(tasks_hard)}")
    if not tasks_easy:
        raise RuntimeError("No tasks available. Run with --generate-tasks to create them.")

    # ── Step 2: Build dataset ──────────────────────────────────────────
    print("\n[2/5] Building GRPO dataset...")
    if args.difficulty == "curriculum":
        # Start with easy only; curriculum will dynamically expand
        initial_tasks = tasks_easy
    elif args.difficulty == "easy":
        initial_tasks = tasks_easy
    elif args.difficulty == "medium":
        initial_tasks = tasks_medium
    else:
        initial_tasks = tasks_hard

    dataset = build_training_dataset(initial_tasks, n_repeats=args.group_size)

    # ── Step 3: Load model ─────────────────────────────────────────────
    print("\n[3/5] Loading model...")
    model, tokenizer = load_model_and_tokenizer(args.model)

    # ── Step 4: Configure GRPO trainer ────────────────────────────────
    print("\n[4/5] Configuring GRPOTrainer...")

    training_args = GRPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=1,
        max_steps=args.steps,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=args.lr,
        max_grad_norm=0.1,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",

        # GRPO specific
        num_generations=args.group_size,
        max_new_tokens=256,
        temperature=0.9,
        top_p=0.95,

        # Logging
        logging_steps=5,
        save_steps=100,
        save_total_limit=2,

        # Memory
        fp16=not is_bfloat16_supported() if UNSLOTH_AVAILABLE else False,
        bf16=is_bfloat16_supported() if UNSLOTH_AVAILABLE else True,
        dataloader_num_workers=0,

        # Reporting
        report_to="none",  # Change to "wandb" if you have it set up
        run_name="chronoveritas-grpo",
        seed=args.seed,

        # Reward
        reward_weights=[1.0],  # Single composite reward
    )

    trainer = GRPOTrainer(
        model=model,
        tokenizer=tokenizer,
        reward_funcs=[reward_fn_wrapper],
        args=training_args,
        train_dataset=dataset,
    )

    # ── Step 5: Train ──────────────────────────────────────────────────
    print(f"\n[5/5] Starting training for {args.steps} steps...")
    print("  Monitor: reward_log.csv in training_logs/")
    print("  Curriculum:", "enabled" if args.difficulty == "curriculum" else "disabled")
    print()

    start_time = time.time()

    if args.difficulty == "curriculum":
        # Phase-based training with dataset refresh
        curriculum = CurriculumManager(
            args.steps,
            {"easy": tasks_easy, "medium": tasks_medium, "hard": tasks_hard}
        )

        phase_steps = {
            "phase1": int(args.steps * 0.37),
            "phase2": int(args.steps * 0.75),
            "phase3": args.steps,
        }

        for phase, (start, end, diffs) in enumerate([
            (0, phase_steps["phase1"], ["easy"]),
            (phase_steps["phase1"], phase_steps["phase2"], ["easy", "medium"]),
            (phase_steps["phase2"], phase_steps["phase3"], ["easy", "medium", "hard"]),
        ]):
            phase_name = f"Phase {phase+1}"
            n_steps = end - start
            print(f"\n{'='*50}")
            print(f"{phase_name}: steps {start}–{end}, difficulties={diffs}")
            print(f"{'='*50}")

            phase_tasks = []
            for d in diffs:
                phase_tasks += {"easy": tasks_easy, "medium": tasks_medium, "hard": tasks_hard}[d]

            if not phase_tasks:
                print(f"  No tasks for {diffs}, skipping phase")
                continue

            phase_dataset = build_training_dataset(phase_tasks, n_repeats=args.group_size)
            trainer.train_dataset = phase_dataset

            # Override max_steps for this phase
            trainer.args.max_steps = end
            trainer.train()
    else:
        trainer.train()

    elapsed = time.time() - start_time
    print(f"\n[Training complete] Elapsed: {elapsed/60:.1f} minutes")

    # ── Save model ─────────────────────────────────────────────────────
    print(f"\n[Saving] → {args.output_dir}")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if UNSLOTH_AVAILABLE:
        # Correct Unsloth save path — do NOT merge 4-bit to 16-bit naively
        model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        print(f"[Saved] LoRA adapters saved (do not merge 4-bit weights directly)")

        # Also save merged 16-bit for inference (correct method)
        merged_dir = args.output_dir + "_merged"
        model.save_pretrained_merged(merged_dir, tokenizer, save_method="merged_16bit")
        print(f"[Saved] Merged 16-bit model → {merged_dir}")
    else:
        model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

    # ── Final log flush ────────────────────────────────────────────────
    _flush_logs(args.steps)
    print(f"[Logs] Reward log → {LOG_DIR}/reward_log.json")
    print(f"[Done] Run `python eval.py --model {args.output_dir}` to generate plots.")


if __name__ == "__main__":
    main()

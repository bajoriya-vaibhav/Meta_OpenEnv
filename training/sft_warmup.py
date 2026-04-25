"""
ChronoVeritas v2 — SFT Warm-Start Script (Phase 0)

Trains the base model on 200 formatted examples to learn output format
before GRPO training. This prevents the cold-start problem where the model
produces no valid JSON for dozens of steps.

Usage:
  python training/sft_warmup.py --n-examples 200 --output ./chronoveritas-sft
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent))

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="unsloth/Qwen2.5-7B-Instruct")
parser.add_argument("--n-examples", type=int, default=200)
parser.add_argument("--output", default="./chronoveritas-sft")
parser.add_argument("--epochs", type=int, default=1)
parser.add_argument("--seed", type=int, default=42)
sft_args = parser.parse_args()

MAX_SEQ_LEN = 1536   # must match train_grpo.py


def build_raw_texts(tasks: List[Dict], n_examples: int = 200) -> List[str]:
    """Return list of fully-formatted text strings (prompt + ideal completion)."""
    from train_grpo import format_single_turn_prompt

    rows: List[str] = []
    for task in tasks:
        prompt = format_single_turn_prompt(task)
        gt = task.get("ground_truth", {})
        completion = json.dumps({
            "verdict":        gt.get("gt_verdict", "false"),
            "mutation_type":  gt.get("gt_mutation_type", "distortion"),
            "mutation_doc_id": gt.get("gt_mutation_doc_id"),
            "provenance_chain": gt.get("gt_provenance_chain", [])[:3],
            "confidence": 0.85,
        }, indent=2)
        rows.append(prompt + completion + "<|im_end|>\n")
        if len(rows) >= n_examples:
            break

    # Repeat to reach n_examples if tasks < n_examples
    while len(rows) < n_examples:
        rows.extend(rows[:n_examples - len(rows)])

    return rows[:n_examples]


def tokenize_dataset(texts: List[str], tokenizer):
    """
    Pre-tokenize all texts with padding + truncation.
    Returns a HuggingFace Dataset with input_ids / attention_mask / labels.
    All tensors are the same length (MAX_SEQ_LEN) — no collator padding needed.
    """
    from datasets import Dataset

    encoded = tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=MAX_SEQ_LEN,
        return_tensors=None,  # return lists, Dataset handles conversion
    )

    # For causal LM, labels == input_ids (shift happens inside the model)
    # Mask padding tokens in labels with -100 so they don't contribute to loss
    labels = []
    for ids, mask in zip(encoded["input_ids"], encoded["attention_mask"]):
        lbl = [id_ if m == 1 else -100 for id_, m in zip(ids, mask)]
        labels.append(lbl)

    return Dataset.from_dict({
        "input_ids":      encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
        "labels":         labels,
    })


def main():
    print("=" * 60)
    print("ChronoVeritas v2 — SFT Warm-Start (Phase 0)")
    print(f"  Model:     {sft_args.model}")
    print(f"  Examples:  {sft_args.n_examples}")
    print(f"  Output:    {sft_args.output}")
    print("=" * 60)

    # ── 1. Generate tasks ──────────────────────────────────────────────────
    from agents.task_bank import SEED_FACTS
    from agents.mutator import Mutator
    from agents.spreader import Spreader
    import random

    mutator = Mutator(seed=sft_args.seed)
    spreader = Spreader(seed=sft_args.seed)
    rng = random.Random(sft_args.seed)
    mutation_types = ["distortion", "fabrication", "omission", "context_shift"]

    tasks: List[Dict] = []
    for i, (diff, n) in enumerate([("easy", 100), ("medium", 60), ("hard", 40)]):
        for j in range(n):
            fact = rng.choice(list(SEED_FACTS))
            mut_type = mutation_types[(i + j) % 4]
            try:
                mutation = mutator.mutate(fact, mutation_type=mut_type)
                task = spreader.spread(mutation, difficulty=diff)
                task["ground_truth"]["corpus_ids"] = [
                    d["doc_id"] for d in task["corpus"]
                ]
                tasks.append(task)
            except Exception as e:
                print(f"  Warning: task generation failed: {e}")

    print(f"[SFT] Generated {len(tasks)} tasks")

    # ── 2. Load model + tokenizer ──────────────────────────────────────────
    try:
        from unsloth import FastLanguageModel
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=sft_args.model,
            max_seq_length=MAX_SEQ_LEN,
            load_in_4bit=True,
        )
        model = FastLanguageModel.get_peft_model(
            model,
            r=8,                    # matches train_grpo.py LoRA rank
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            lora_alpha=16,
            lora_dropout=0,
            bias="none",
            use_gradient_checkpointing="unsloth",
        )
    except ImportError:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        model = AutoModelForCausalLM.from_pretrained(
            sft_args.model, torch_dtype=torch.bfloat16, device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(sft_args.model)

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"   # right-pad for causal LM

    # ── 3. Build pre-tokenized dataset ────────────────────────────────────
    # Pre-tokenize NOW with padding + truncation so all sequences are the
    # same length (MAX_SEQ_LEN). The trainer/collator never sees raw text
    # strings again, eliminating all "Unable to create tensor" errors.
    raw_texts = build_raw_texts(tasks, sft_args.n_examples)
    dataset = tokenize_dataset(raw_texts, tokenizer)
    print(f"[SFT] Dataset: {len(dataset)} pre-tokenized examples "
          f"(seq_len={MAX_SEQ_LEN})")

    # ── 4. Train ──────────────────────────────────────────────────────────
    from transformers import TrainingArguments, DataCollatorForLanguageModeling

    # Use base TrainingArguments (not SFTConfig) since we handle tokenization
    # ourselves. DataCollatorForLanguageModeling with mlm=False handles the
    # causal-LM loss masking on already-tokenized data.
    training_args = TrainingArguments(
        output_dir=sft_args.output,
        num_train_epochs=sft_args.epochs,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        warmup_ratio=0.05,
        logging_steps=10,
        save_steps=50,
        fp16=False,
        bf16=True,
        seed=sft_args.seed,
        remove_unused_columns=False,   # keep all columns (input_ids, labels, etc.)
        report_to="none",
        dataloader_num_workers=0,
    )

    # DataCollatorForLanguageModeling with mlm=False:
    # - does NOT pad (already padded to max_length)
    # - does NOT modify input_ids / labels (we set labels=-100 for padding above)
    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,          # causal LM, not masked LM
        pad_to_multiple_of=None,
    )

    from transformers import Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
        tokenizer=tokenizer,
    )

    trainer.train()

    # ── 5. Save ───────────────────────────────────────────────────────────
    model.save_pretrained(sft_args.output)
    tokenizer.save_pretrained(sft_args.output)
    print(f"\n[SFT] ✓ Model saved to {sft_args.output}")
    print("[SFT] Now run: python train_grpo.py --model ./chronoveritas-sft ...")


if __name__ == "__main__":
    main()

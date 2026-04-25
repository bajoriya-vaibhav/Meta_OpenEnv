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


def build_sft_dataset(tasks: List[Dict], n_examples: int = 200):
    """Build (prompt, correct_completion) pairs from tasks."""
    from datasets import Dataset
    from train_grpo import format_single_turn_prompt

    rows = []
    for task in tasks:
        prompt = format_single_turn_prompt(task)
        gt = task.get("ground_truth", {})

        # Build ideal completion
        completion = json.dumps({
            "verdict": gt.get("gt_verdict", "false"),
            "mutation_type": gt.get("gt_mutation_type", "distortion"),
            "mutation_doc_id": gt.get("gt_mutation_doc_id"),
            "provenance_chain": gt.get("gt_provenance_chain", [])[:3],
            "confidence": 0.85,
        }, indent=2)

        rows.append({
            "text": prompt + completion + "<|im_end|>\n"
        })

        if len(rows) >= n_examples:
            break

    # Repeat if we need more examples
    while len(rows) < n_examples:
        rows.extend(rows[:n_examples - len(rows)])

    return Dataset.from_list(rows[:n_examples])


def main():
    print("=" * 60)
    print("ChronoVeritas v2 — SFT Warm-Start (Phase 0)")
    print(f"  Model:     {sft_args.model}")
    print(f"  Examples:  {sft_args.n_examples}")
    print(f"  Output:    {sft_args.output}")
    print("=" * 60)

    # Generate tasks for SFT
    from agents.task_bank import SEED_FACTS
    from agents.mutator import Mutator
    from agents.spreader import Spreader

    mutator = Mutator(seed=sft_args.seed)
    spreader = Spreader(seed=sft_args.seed)
    import random
    rng = random.Random(sft_args.seed)
    mutation_types = ["distortion", "fabrication", "omission", "context_shift"]

    tasks = []
    for i, (diff, n) in enumerate([("easy", 100), ("medium", 60), ("hard", 40)]):
        for j in range(n):
            fact = rng.choice(list(SEED_FACTS))
            mut_type = mutation_types[(i + j) % 4]
            try:
                mutation = mutator.mutate(fact, mutation_type=mut_type)
                task = spreader.spread(mutation, difficulty=diff)
                task["ground_truth"]["corpus_ids"] = [d["doc_id"] for d in task["corpus"]]
                tasks.append(task)
            except Exception as e:
                print(f"  Warning: task generation failed: {e}")

    print(f"[SFT] Generated {len(tasks)} tasks for SFT dataset")

    dataset = build_sft_dataset(tasks, sft_args.n_examples)
    print(f"[SFT] Dataset: {len(dataset)} examples")

    # Load model
    try:
        from unsloth import FastLanguageModel
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=sft_args.model,
            max_seq_length=2048,
            load_in_4bit=True,
        )
        model = FastLanguageModel.get_peft_model(
            model, r=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            lora_alpha=16, lora_dropout=0, bias="none",
            use_gradient_checkpointing="unsloth",
        )
    except ImportError:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        model = AutoModelForCausalLM.from_pretrained(
            sft_args.model, torch_dtype=torch.bfloat16, device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(sft_args.model)

    tokenizer.pad_token = tokenizer.eos_token

    # Train
    from transformers import TrainingArguments, Trainer

    training_args = TrainingArguments(
        output_dir=sft_args.output,
        num_train_epochs=sft_args.epochs,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        warmup_ratio=0.05,
        logging_steps=10,
        save_steps=50,
        fp16=False,
        bf16=True,
        seed=sft_args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )

    trainer.train()

    # Save
    model.save_pretrained(sft_args.output)
    tokenizer.save_pretrained(sft_args.output)
    print(f"[SFT] Model saved to {sft_args.output}")


if __name__ == "__main__":
    main()

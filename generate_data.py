"""
ChronoVeritas v2 — Standalone Data Generator

Generates adversarial tasks using the Mutator and Spreader without
requiring any GPU training libraries (TRL, Unsloth, Transformers).
"""
import argparse
import json
import random
from pathlib import Path
from agents.task_bank import SEED_FACTS
from agents.mutator import Mutator
from agents.spreader import Spreader

def main():
    parser = argparse.ArgumentParser(description="Generate tasks for ChronoVeritas v2")
    parser.add_argument("--easy", type=int, default=10, help="Number of easy tasks")
    parser.add_argument("--medium", type=int, default=10, help="Number of medium tasks")
    parser.add_argument("--hard", type=int, default=5, help="Number of hard tasks")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output-dir", type=str, default="data/tasks/generated", help="Output directory")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mutator = Mutator(seed=args.seed)
    spreader = Spreader(seed=args.seed)
    rng = random.Random(args.seed)
    
    mutation_types = ["distortion", "fabrication", "omission", "context_shift"]
    total_generated = 0

    print(f"Generating tasks to {out_dir}/ ...")

    for diff, count in [("easy", args.easy), ("medium", args.medium), ("hard", args.hard)]:
        for i in range(count):
            # Pick a random seed fact
            fact = rng.choice(list(SEED_FACTS))
            # Cycle through mutation types to ensure a balanced dataset
            mut_type = mutation_types[(total_generated + i) % 4]
            
            try:
                mutation = mutator.mutate(fact, mutation_type=mut_type)
                task = spreader.spread(mutation, difficulty=diff)
                
                # Enrich ground truth for training scripts
                task["ground_truth"]["corpus_ids"] = [d["doc_id"] for d in task["corpus"]]
                
                # Save to disk
                file_path = out_dir / f"{task['task_id']}.json"
                with open(file_path, "w") as f:
                    json.dump(task, f, indent=2)
                
                print(f"  [{diff.upper()}] {task['task_id']} — {mut_type} applied to {fact.fact_id}")
                total_generated += 1
            except Exception as e:
                print(f"  [ERROR] Failed to generate task: {e}")

    print(f"\n✅ Successfully generated {total_generated} tasks in {out_dir}/")

if __name__ == "__main__":
    main()

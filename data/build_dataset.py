"""
ChronoVeritas — Build / verify dataset.
Generates the minimal task JSON files and corpus data.
"""
from __future__ import annotations

import json
import os
from pathlib import Path


TASKS_DIR = Path(__file__).resolve().parent / "tasks"
CORPUS_DIR = Path(__file__).resolve().parent / "corpus"


def build_corpus_files() -> None:
    """
    Extract corpus documents from task files and write them as individual
    JSON files in data/corpus/ for alternative loading patterns.
    """
    CORPUS_DIR.mkdir(exist_ok=True)

    for task_path in sorted(TASKS_DIR.glob("*.json")):
        with open(task_path) as f:
            task = json.load(f)

        for doc in task.get("corpus", []):
            doc_path = CORPUS_DIR / f"{doc['doc_id']}.json"
            with open(doc_path, "w") as f:
                json.dump(doc, f, indent=2)
            print(f"  Wrote {doc_path}")


def verify_tasks() -> None:
    """Verify all task files have required fields."""
    required_keys = {"task_id", "difficulty", "max_steps", "claim", "ground_truth", "corpus"}
    gt_keys = {"gt_verdict", "gt_mutation_type", "gt_mutation_doc_id", "gt_provenance_chain", "gt_timeline"}

    for task_path in sorted(TASKS_DIR.glob("*.json")):
        with open(task_path) as f:
            task = json.load(f)

        missing = required_keys - set(task.keys())
        if missing:
            print(f"  [FAIL] {task_path.name}: missing keys: {missing}")
            continue

        gt_missing = gt_keys - set(task["ground_truth"].keys())
        if gt_missing:
            print(f"  [FAIL] {task_path.name}: ground_truth missing: {gt_missing}")
            continue

        n_docs = len(task["corpus"])
        print(f"  [OK] {task_path.name}: task_id={task['task_id']}, "
              f"difficulty={task['difficulty']}, {n_docs} docs")


if __name__ == "__main__":
    print("Verifying tasks...")
    verify_tasks()
    print("\nBuilding corpus files...")
    build_corpus_files()
    print("\nDone!")

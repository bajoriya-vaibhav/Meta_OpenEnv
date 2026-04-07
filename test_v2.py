"""Verify tasks parse and show key info."""
import json
from env.models import TaskSpec

for fname in ['task_easy.json', 'task_medium.json', 'task_hard.json']:
    with open(f'data/tasks/{fname}') as f:
        task = TaskSpec(**json.load(f))
    print(f"{task.task_id} ({task.difficulty}) — {len(task.corpus)} docs")
    print(f"  verdict={task.ground_truth.gt_verdict}, "
          f"type={task.ground_truth.gt_mutation_type}, "
          f"doc={task.ground_truth.gt_mutation_doc_id}")
    print(f"  chain={task.ground_truth.gt_provenance_chain}")
    if task.ground_truth.gt_conflict_fields:
        print(f"  conflicts={task.ground_truth.gt_conflict_fields}")
    for doc in task.corpus:
        print(f"    {doc.doc_id}: T{doc.reliability_tier} {doc.source}")
    print()

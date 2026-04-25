"""
ChronoVeritas v2 — Reward Function Module

Standalone module containing compute_reward() and extract_json_safe()
for use by both train_grpo.py and eval.py.

This is a re-export of the functions defined in train_grpo.py,
provided here for clean import paths.
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional, Tuple


def extract_json_safe(text: str) -> Optional[Dict]:
    """
    Robustly extract JSON from model output.
    Tries multiple patterns in order of specificity.
    """
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
    5-component proxy reward for GRPO training.

    Components:
      - format:              +0.05  (valid JSON with required fields)
      - verdict:             +0.35  (correct verdict)
      - mutation_type:       +0.25  (correct mutation type)
      - mutation_point:      +0.25  (correct mutation doc, 0.12 for adjacent)
      - calibration:         +0.05  (Brier-based confidence calibration)
      - hallucination_penalty: -0.05 per fabricated doc_id (capped at -0.20)

    Total clamped to [-0.20, 1.0].
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
        mp_score = 0.12
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

    # ── Hallucination penalty ──────────────────────────────────────────
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

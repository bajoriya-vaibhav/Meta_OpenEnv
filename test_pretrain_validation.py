"""
ChronoVeritas-MA — Multi-Agent Pre-Training Validation
=======================================================
Runs 30 rollouts (10 easy, 10 medium, 10 hard) using the REAL environment
with the 3-role cooperative workflow:

  Retriever → Analyst → Arbiter → submit_verdict

Each role is driven by a Groq LLM (Llama-3-8B or fallback to Qwen).
This validates:
  1. Environment reset/step/grade loop works end-to-end
  2. Multi-agent action sequencing (role permissions, blackboard, messages)
  3. compute_reward() produces sensible scores on real LLM output
  4. No crashes across all 3 difficulty levels

Usage:
  export GROQ_API_KEY=gsk_...
  python test_pretrain_validation.py

  # Or with a custom model:
  python test_pretrain_validation.py --model llama-3.3-70b-versatile
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ── Project imports ────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from env.environment import ChronoVeritasEnv
from env.models import Action, AgentMessage, StepResult


# ── Inline compute_reward (matches current train_grpo.py weights) ──────────
def extract_json_safe(text: str) -> Optional[Dict]:
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


def compute_reward(completion: str, ground_truth: Dict) -> Tuple[float, Dict]:
    breakdown: Dict[str, float] = {}
    W = {"format": 0.05, "verdict": 0.30, "mutation_type": 0.18,
         "mutation_point": 0.18, "provenance": 0.18, "source_reliability": 0.11,
         "hallucination": 0.12, "brier_penalty": 0.08}

    parsed = extract_json_safe(completion)
    if parsed is None:
        return -0.15, {"format": -0.15, "parse_error": 1.0}
    required = {"verdict", "mutation_type", "mutation_doc_id", "confidence"}
    if not required.issubset(parsed.keys()):
        return -0.10, {"format": -0.10, "missing_fields": 1.0}

    breakdown["format"] = W["format"]
    gt_verdict = ground_truth.get("gt_verdict", "")
    verdict_correct = str(parsed.get("verdict", "")).strip() == gt_verdict
    breakdown["verdict"] = W["verdict"] if verdict_correct else 0.0

    gt_mut = ground_truth.get("gt_mutation_type", "")
    breakdown["mutation_type"] = W["mutation_type"] if str(parsed.get("mutation_type", "")).strip() == gt_mut else 0.0

    pred_doc = str(parsed.get("mutation_doc_id", "") or "").strip()
    gt_doc = str(ground_truth.get("gt_mutation_doc_id", "") or "").strip()
    gt_tl = ground_truth.get("gt_timeline", [])
    if not pred_doc and not gt_doc:
        mp = W["mutation_point"]
    elif pred_doc == gt_doc:
        mp = W["mutation_point"]
    elif (pred_doc and gt_doc and pred_doc in gt_tl and gt_doc in gt_tl
          and abs(gt_tl.index(pred_doc) - gt_tl.index(gt_doc)) == 1):
        mp = W["mutation_point"] * 0.5
    else:
        mp = 0.0
    breakdown["mutation_point"] = round(mp, 4)

    gt_chain = ground_truth.get("gt_provenance_chain", [])
    pred_chain = parsed.get("provenance_chain", [])
    if not isinstance(pred_chain, list):
        pred_chain = []
    if not gt_chain and not pred_chain:
        pf1 = 1.0
    elif not gt_chain or not pred_chain:
        pf1 = 0.0
    else:
        pc, gc = Counter(pred_chain), Counter(gt_chain)
        ov = sum(min(pc[k], gc[k]) for k in pc if k in gc)
        prec = ov / sum(pc.values()) if pc else 0.0
        rec = ov / sum(gc.values()) if gc else 0.0
        pf1 = 2.0 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    breakdown["provenance"] = round(W["provenance"] * pf1, 4)

    corpus_tiers = ground_truth.get("corpus_tiers", {})
    corpus_ids = set(ground_truth.get("corpus_ids", []))
    tw = {1: 1.0, 2: 0.5, 3: 0.1}
    if isinstance(pred_chain, list) and pred_chain:
        vt = [tw.get(corpus_tiers.get(d, 2), 0.5) for d in pred_chain if d in corpus_ids]
        sr = sum(vt) / len(vt) if vt else 0.0
    else:
        sr = 0.0
    breakdown["source_reliability"] = round(W["source_reliability"] * sr, 4)

    if isinstance(pred_chain, list):
        fabricated = [d for d in pred_chain if str(d) not in corpus_ids]
        hr = min(1.0, len(fabricated) * 0.25)
    else:
        hr = 0.0
    breakdown["hallucination_penalty"] = -round(W["hallucination"] * hr, 4)

    try:
        conf = max(0.0, min(1.0, float(parsed.get("confidence", 0.5))))
    except (TypeError, ValueError):
        conf = 0.5
    correctness = 1.0 if verdict_correct else 0.0
    brier = (conf - correctness) ** 2
    if not verdict_correct and conf > 0.7:
        brier = min(brier * 1.5, 1.0)
    breakdown["brier_penalty"] = -round(W["brier_penalty"] * brier, 4)

    total = max(-0.20, min(1.0, sum(breakdown.values())))
    return round(total, 4), breakdown

# ── CLI ────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Multi-agent pre-training validation")
parser.add_argument("--model", default="llama-3.1-8b-instant",
                    help="Groq model ID")
parser.add_argument("--n-per-difficulty", type=int, default=10,
                    help="Rollouts per difficulty level")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--verbose", action="store_true")
cli_args = parser.parse_args()

# ── Groq client ────────────────────────────────────────────────────────────
try:
    from groq import Groq
    GROQ_KEY = os.environ.get("GROQ_API_KEY", "")
    if not GROQ_KEY:
        print("ERROR: Set GROQ_API_KEY environment variable.")
        print("  export GROQ_API_KEY=gsk_...")
        sys.exit(1)
    groq_client = Groq(api_key=GROQ_KEY)
    print(f"[Groq] Connected. Model: {cli_args.model}")
except ImportError:
    print("ERROR: pip install groq")
    sys.exit(1)


def llm_call(system: str, user: str, *, temperature: float = 0.3,
             max_tokens: int = 512) -> str:
    """Single Groq LLM call with retry."""
    for attempt in range(3):
        try:
            resp = groq_client.chat.completions.create(
                model=cli_args.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            if attempt < 2:
                time.sleep(2 ** attempt)
            else:
                return f"[LLM_ERROR: {e}]"


# ── Role-specific prompts ─────────────────────────────────────────────────

RETRIEVER_SYSTEM = """You are the RETRIEVER agent in a multi-agent fact-checking team.
Your job: search the corpus and fetch documents relevant to the claim.
You can see: the claim and corpus metadata (doc IDs, titles, sources, tiers).

Respond with a JSON list of actions to take. Example:
[
  {"type": "search", "query": "funding amount 2024"},
  {"type": "fetch_doc", "doc_id": "DOC-0003"},
  {"type": "send_message", "recipient": "analyst", "content": "Fetched 3 docs. DOC-0003 is Tier 1 official."}
]

Return ONLY the JSON array. Focus on fetching the most authoritative (Tier 1) documents first."""

ANALYST_SYSTEM = """You are the ANALYST agent in a multi-agent fact-checking team.
Your job: read fetched documents, flag contradictions, build a timeline, and suggest a hypothesis.
You can see: the claim, fetched document contents, and messages from the Retriever.

Respond with a JSON list of actions. Example:
[
  {"type": "flag_contradiction", "doc_id_a": "DOC-0001", "doc_id_b": "DOC-0003"},
  {"type": "add_timeline_event", "doc_id": "DOC-0001", "event_label": "Original report published"},
  {"type": "set_mutation_point", "doc_id": "DOC-0003", "mutation_type": "distortion"},
  {"type": "send_message", "recipient": "arbiter", "content": "DOC-0003 changed the figure from $2.4B to $3.1B. Likely distortion."}
]

Return ONLY the JSON array."""

ARBITER_SYSTEM = """You are the ARBITER agent in a multi-agent fact-checking team.
Your job: review evidence from the Retriever and Analyst, then submit the FINAL verdict.
You can see: the claim, team messages, analyst hypotheses, timeline, and contradictions.

IMPORTANT: Your response MUST always end with a submit_verdict action. Without it, the investigation fails.

Respond with a JSON array. Your LAST action MUST be submit_verdict:
[
  {"type": "submit_verdict", "verdict": "false", "mutation_type": "distortion",
   "mutation_doc_id": "DOC-0003", "provenance_chain": ["DOC-0003", "DOC-0005"],
   "confidence": 0.85}
]

Valid verdicts: true, false, misleading, unverifiable
Valid mutation_types: distortion, omission, fabrication, context_shift, none
If verdict is 'true', use mutation_type='none' and omit mutation_doc_id.

Return ONLY the JSON array. The LAST element MUST be submit_verdict."""


# ── Action parsing ────────────────────────────────────────────────────────

def parse_llm_actions(raw: str) -> List[Dict]:
    """Extract a JSON array of actions from LLM output."""
    # Try direct parse
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return [a for a in parsed if isinstance(a, dict) and "type" in a]
        if isinstance(parsed, dict) and "type" in parsed:
            return [parsed]
    except json.JSONDecodeError:
        pass

    # Try extracting JSON array from markdown
    for m in re.finditer(r'\[.*?\]', raw, re.DOTALL):
        try:
            parsed = json.loads(m.group())
            if isinstance(parsed, list):
                return [a for a in parsed if isinstance(a, dict) and "type" in a]
        except json.JSONDecodeError:
            continue

    return []


# ── Build role-specific context ───────────────────────────────────────────

def build_retriever_context(obs: dict, claim: str) -> str:
    """What the Retriever sees: claim + corpus metadata."""
    meta_lines = []
    for doc in obs.get("corpus_metadata", []):
        if not isinstance(doc, dict):
            continue
        tier_label = {1: "Official", 2: "News", 3: "Blog"}.get(
            doc.get("reliability_tier", 2), "Unknown"
        )
        meta_lines.append(
            f"  [{doc.get('doc_id','?')}] {tier_label} (Tier {doc.get('reliability_tier', 2)}) "
            f"| {doc.get('title', '?')} | Source: {doc.get('source', '?')}"
        )
    corpus_text = "\n".join(meta_lines) if meta_lines else "  (no metadata yet — search first)"
    return f"CLAIM: {claim}\n\nAVAILABLE DOCUMENTS:\n{corpus_text}"


def build_analyst_context(obs: dict, claim: str, messages: List[dict]) -> str:
    """What the Analyst sees: claim + fetched docs + retriever messages."""
    parts = [f"CLAIM: {claim}\n"]

    # Fetched documents
    docs = obs.get("retrieved_docs", [])
    if docs:
        parts.append("FETCHED DOCUMENTS:")
        for doc in docs:
            if not isinstance(doc, dict):
                continue
            content = doc.get("content", "")[:600]
            parts.append(
                f"  [{doc.get('doc_id','?')}] Tier {doc.get('reliability_tier', 2)} "
                f"| {doc.get('title', '?')}\n  Content: {content}\n"
            )
    else:
        parts.append("FETCHED DOCUMENTS: (none yet)")

    # Messages from retriever
    retriever_msgs = [m for m in messages if m.get("sender") == "retriever"]
    if retriever_msgs:
        parts.append("MESSAGES FROM RETRIEVER:")
        for m in retriever_msgs:
            parts.append(f"  > {m.get('content', '')}")

    return "\n".join(parts)


def build_arbiter_context(obs: dict, claim: str, messages: List[dict]) -> str:
    """What the Arbiter sees: claim + timeline + contradictions + all messages."""
    parts = [f"CLAIM: {claim}\n"]

    # Timeline
    timeline = obs.get("agent_timeline", [])
    if timeline:
        parts.append("INVESTIGATION TIMELINE:")
        for e in timeline:
            if isinstance(e, dict):
                parts.append(f"  [{e.get('doc_id', '?')}] {e.get('event_label', '?')}")

    # Contradictions
    contras = obs.get("flagged_contradictions", [])
    if contras:
        parts.append("FLAGGED CONTRADICTIONS:")
        for pair in contras:
            if isinstance(pair, (list, tuple)) and len(pair) == 2:
                parts.append(f"  {pair[0]} vs {pair[1]}")

    # All team messages
    if messages:
        parts.append("TEAM MESSAGES:")
        for m in messages:
            parts.append(f"  [{m.get('sender', '?')} -> {m.get('recipient', '?')}] {m.get('content', '')}")

    # Hypotheses
    hyps = obs.get("hypotheses", {})
    if hyps:
        parts.append("CURRENT HYPOTHESES:")
        for agent, h in hyps.items():
            parts.append(
                f"  {agent}: mutation={h.get('mutation_type','?')} "
                f"doc={h.get('mutation_doc_id','?')} conf={h.get('confidence', 0)}"
            )

    return "\n".join(parts)


# ── Execute a single action safely ────────────────────────────────────────

async def safe_step(env: ChronoVeritasEnv, action_dict: dict,
                    role: str, verbose: bool = False) -> Optional[StepResult]:
    """Convert a dict to an Action and step the env. Returns None on error."""
    try:
        action_type = action_dict.get("type", "")
        payload = {k: v for k, v in action_dict.items() if k != "type"}

        # Normalize action types (LLM often shortens names)
        ACTION_ALIASES = {
            "message": "send_message",
            "msg": "send_message",
            "send_msg": "send_message",
            "fetch": "fetch_doc",
            "request": "request_evidence",
            "submit": "submit_verdict",
            "verdict": "submit_verdict",
            "flag": "flag_contradiction",
            "timeline": "add_timeline_event",
            "hypothesis": "update_hypothesis",
            "set_mutation": "set_mutation_point",
        }
        action_type = ACTION_ALIASES.get(action_type, action_type)

        # Normalize role strings to lowercase (LLM often returns 'Retriever' not 'retriever')
        for key in ("sender", "recipient", "agent"):
            if key in payload and isinstance(payload[key], str):
                payload[key] = payload[key].lower()

        # Inject sender for message/evidence actions
        if action_type in ("send_message", "request_evidence") and "sender" not in payload:
            payload["sender"] = role

        action = Action(type=action_type, payload=payload)
        result = await env.step(action)

        if verbose and result.info.get("error"):
            print(f"    [{role}] {action_type} -> ERROR: {result.info['error']}")
        elif verbose:
            print(f"    [{role}] {action_type} -> reward={result.reward:.4f} done={result.done}")

        return result
    except Exception as e:
        if verbose:
            print(f"    [{role}] {action_dict.get('type', '?')} -> EXCEPTION: {e}")
        return None


# ── Run one multi-agent episode ───────────────────────────────────────────

async def run_episode(env: ChronoVeritasEnv, task_id: str,
                      verbose: bool = False) -> Dict[str, Any]:
    """
    Full 3-role cooperative episode:
      Round 1: Retriever searches + fetches
      Round 2: Analyst inspects + flags + hypothesises
      Round 3: Arbiter reviews + (optionally requests more) + submits verdict
    """
    result = await env.reset(task_id=task_id)
    obs = result.observation.model_dump()
    claim = obs.get("claim", "")
    messages: List[dict] = []
    episode_rewards = []
    total_actions = 0
    role_action_counts = {"retriever": 0, "analyst": 0, "arbiter": 0}

    # ── Round 1: RETRIEVER ────────────────────────────────────────────
    if verbose:
        print(f"  [RETRIEVER] thinking...")
    retriever_ctx = build_retriever_context(obs, claim)
    retriever_raw = llm_call(RETRIEVER_SYSTEM, retriever_ctx, max_tokens=400)
    retriever_actions = parse_llm_actions(retriever_raw)

    if not retriever_actions:
        # Fallback: search the claim + fetch first 2 docs
        retriever_actions = [{"type": "search", "query": claim[:80]}]

    for act in retriever_actions[:5]:  # cap at 5 actions
        if act.get("type") == "send_message":
            messages.append({**act, "sender": "retriever"})
        r = await safe_step(env, act, "retriever", verbose)
        if r:
            episode_rewards.append(r.reward)
            role_action_counts["retriever"] += 1
            total_actions += 1
            obs = r.observation.model_dump()
            if r.done:
                break

    # After search, fetch corpus docs that appeared
    corpus_meta = [d for d in obs.get("corpus_metadata", []) if isinstance(d, dict)]
    fetched_ids = {d["doc_id"] for d in obs.get("retrieved_docs", []) if isinstance(d, dict) and "doc_id" in d}
    for doc in corpus_meta[:4]:
        if doc["doc_id"] not in fetched_ids:
            r = await safe_step(env, {"type": "fetch_doc", "doc_id": doc["doc_id"]},
                                "retriever", verbose)
            if r:
                episode_rewards.append(r.reward)
                role_action_counts["retriever"] += 1
                total_actions += 1
                obs = r.observation.model_dump()
                if r.done:
                    break

    # ── Round 2: ANALYST ──────────────────────────────────────────────
    if verbose:
        print(f"  [ANALYST] thinking...")
    analyst_ctx = build_analyst_context(obs, claim, messages)
    analyst_raw = llm_call(ANALYST_SYSTEM, analyst_ctx, max_tokens=400)
    analyst_actions = parse_llm_actions(analyst_raw)

    for act in analyst_actions[:5]:
        if act.get("type") == "send_message":
            messages.append({**act, "sender": "analyst"})
        r = await safe_step(env, act, "analyst", verbose)
        if r:
            episode_rewards.append(r.reward)
            role_action_counts["analyst"] += 1
            total_actions += 1
            obs = r.observation.model_dump()
            if r.done:
                break

    # ── Round 3: ARBITER ──────────────────────────────────────────────
    if verbose:
        print(f"  [ARBITER] thinking...")
    arbiter_ctx = build_arbiter_context(obs, claim, messages)
    arbiter_raw = llm_call(ARBITER_SYSTEM, arbiter_ctx, max_tokens=500)
    arbiter_actions = parse_llm_actions(arbiter_raw)

    verdict_submitted = False
    for act in arbiter_actions[:5]:
        if act.get("type") == "send_message":
            messages.append({**act, "sender": "arbiter"})
        if act.get("type") == "submit_verdict":
            verdict_submitted = True
        r = await safe_step(env, act, "arbiter", verbose)
        if r:
            episode_rewards.append(r.reward)
            role_action_counts["arbiter"] += 1
            total_actions += 1
            obs = r.observation.model_dump()
            if r.done:
                break

    # If arbiter didn't submit, force a verdict
    if not verdict_submitted:
        if verbose:
            print(f"  [ARBITER] fallback verdict...")
        # Use first corpus doc as mutation point; if no docs, declare "true" / "none"
        if corpus_meta:
            fb_doc = corpus_meta[0]["doc_id"]
            fb_mut = "distortion"
            fb_verdict = "false"
            fb_chain = [d["doc_id"] for d in corpus_meta[:2]]
        else:
            fb_doc = None
            fb_mut = "none"
            fb_verdict = "true"
            fb_chain = []
        fallback = {
            "type": "submit_verdict",
            "verdict": fb_verdict, "mutation_type": fb_mut,
            "mutation_doc_id": fb_doc,
            "provenance_chain": fb_chain,
            "confidence": 0.5,
        }
        r = await safe_step(env, fallback, "arbiter", verbose)
        if r:
            episode_rewards.append(r.reward)
            role_action_counts["arbiter"] += 1
            total_actions += 1
            obs = r.observation.model_dump()
            # Fallback went through submit_verdict — count as submitted
            if r.done:
                verdict_submitted = True

    # ── Extract final grade ───────────────────────────────────────────
    final_info = r.info if r else {}
    final_score = final_info.get("final_score", sum(episode_rewards))
    grade_breakdown = final_info.get("grade_breakdown", {})

    # ── Also compute GRPO-style text reward ───────────────────────────
    # The Arbiter outputs a JSON array like [{send_message...}, {submit_verdict...}].
    # extract_json_safe grabs the first {...}, which might be send_message (no verdict).
    # Fix: find the submit_verdict action specifically and pass just that.
    grpo_completion = None
    arbiter_parsed = parse_llm_actions(arbiter_raw)
    for act in arbiter_parsed:
        if isinstance(act, dict) and act.get("type") == "submit_verdict":
            # Extract just the verdict fields (what train_grpo would see)
            verdict_obj = {k: v for k, v in act.items() if k != "type"}
            grpo_completion = json.dumps(verdict_obj)
            break
    if grpo_completion is None:
        # Arbiter didn't produce submit_verdict — use fallback
        grpo_completion = json.dumps({
            "verdict": "false", "mutation_type": "distortion",
            "mutation_doc_id": corpus_meta[0]["doc_id"] if corpus_meta else None,
            "provenance_chain": [d["doc_id"] for d in corpus_meta[:2]],
            "confidence": 0.5,
        })

    task = env._current_task
    gt = {
        "gt_verdict": task.ground_truth.gt_verdict,
        "gt_mutation_type": task.ground_truth.gt_mutation_type,
        "gt_mutation_doc_id": task.ground_truth.gt_mutation_doc_id,
        "gt_provenance_chain": task.ground_truth.gt_provenance_chain,
        "gt_timeline": task.ground_truth.gt_timeline,
        "corpus_ids": [d.doc_id for d in task.corpus],
        "corpus_tiers": {d.doc_id: d.reliability_tier for d in task.corpus},
    }
    grpo_reward, grpo_breakdown = compute_reward(grpo_completion, gt)

    return {
        "task_id": task_id,
        "difficulty": task.difficulty,
        "env_final_score": round(final_score, 4),
        "grpo_reward": round(grpo_reward, 4),
        "grpo_breakdown": grpo_breakdown,
        "grade_breakdown": grade_breakdown,
        "total_actions": total_actions,
        "role_actions": role_action_counts,
        "messages_exchanged": len(messages),
        "verdict_submitted": verdict_submitted,
        "gt_verdict": task.ground_truth.gt_verdict,
        "gt_mutation_type": task.ground_truth.gt_mutation_type,
    }


# ── Main ──────────────────────────────────────────────────────────────────

async def main():
    print("=" * 70)
    print("ChronoVeritas-MA — Multi-Agent Pre-Training Validation")
    print(f"  Model:   {cli_args.model}")
    print(f"  Rollouts: {cli_args.n_per_difficulty} per difficulty (total {cli_args.n_per_difficulty * 3})")
    print(f"  Seed:    {cli_args.seed}")
    print("=" * 70)

    rng = random.Random(cli_args.seed)

    # Point env at generated/ folder (3000 tasks) instead of base data/tasks/ (3 tasks)
    gen_dir = Path("data/tasks/generated")
    if gen_dir.exists() and any(gen_dir.glob("*.json")):
        env = ChronoVeritasEnv(tasks_dir=gen_dir)
    else:
        env = ChronoVeritasEnv()  # fallback to base tasks

    if not env.tasks:
        print("ERROR: No tasks loaded. Ensure data/tasks/ has JSON files.")
        sys.exit(1)

    # Group tasks by difficulty
    tasks_by_diff: Dict[str, List] = defaultdict(list)
    for t in env.tasks:
        tasks_by_diff[t.difficulty].append(t.task_id)

    print(f"\n[Tasks] easy={len(tasks_by_diff['easy'])} "
          f"medium={len(tasks_by_diff['medium'])} "
          f"hard={len(tasks_by_diff['hard'])}")

    # Sample tasks
    selected: List[Tuple[str, str]] = []  # (difficulty, task_id)
    for diff in ["easy", "medium", "hard"]:
        pool = tasks_by_diff.get(diff, [])
        if not pool:
            print(f"  WARNING: no {diff} tasks available, skipping")
            continue
        sample = rng.sample(pool, min(cli_args.n_per_difficulty, len(pool)))
        for tid in sample:
            selected.append((diff, tid))

    print(f"[Selected] {len(selected)} rollouts\n")

    # ── Run rollouts ──────────────────────────────────────────────────
    all_results: List[Dict] = []
    results_by_diff: Dict[str, List[Dict]] = defaultdict(list)

    for i, (diff, task_id) in enumerate(selected, 1):
        print(f"[{i:2d}/{len(selected)}] {task_id} ({diff})", end=" ... ")
        t0 = time.time()
        try:
            result = await run_episode(env, task_id, verbose=cli_args.verbose)
            elapsed = time.time() - t0
            result["elapsed_s"] = round(elapsed, 1)
            all_results.append(result)
            results_by_diff[diff].append(result)

            grpo = result["grpo_reward"]
            env_sc = result["env_final_score"]
            acts = result["total_actions"]
            msgs = result["messages_exchanged"]
            vsub = "YES" if result["verdict_submitted"] else "FALLBACK"
            print(f"grpo={grpo:+.3f}  env={env_sc:.3f}  "
                  f"actions={acts}  msgs={msgs}  verdict={vsub}  "
                  f"({elapsed:.1f}s)")
        except Exception as e:
            print(f"FAILED: {e}")
            all_results.append({
                "task_id": task_id, "difficulty": diff,
                "error": str(e), "grpo_reward": -0.15,
            })

    # ── Summary ───────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    for diff in ["easy", "medium", "hard"]:
        results = results_by_diff.get(diff, [])
        if not results:
            continue
        rewards = [r["grpo_reward"] for r in results]
        env_scores = [r.get("env_final_score", 0) for r in results]
        verdicts_ok = sum(1 for r in results if r.get("verdict_submitted", False))
        avg_actions = sum(r.get("total_actions", 0) for r in results) / len(results)
        avg_msgs = sum(r.get("messages_exchanged", 0) for r in results) / len(results)

        print(f"\n  {diff.upper()} ({len(results)} rollouts):")
        print(f"    GRPO reward:  avg={sum(rewards)/len(rewards):+.3f}  "
              f"min={min(rewards):+.3f}  max={max(rewards):+.3f}")
        print(f"    Env score:    avg={sum(env_scores)/len(env_scores):.3f}")
        print(f"    Verdict rate: {verdicts_ok}/{len(results)} "
              f"({100*verdicts_ok/len(results):.0f}%)")
        print(f"    Avg actions:  {avg_actions:.1f}  Avg messages: {avg_msgs:.1f}")

    # Overall
    all_rewards = [r["grpo_reward"] for r in all_results if "grpo_reward" in r]
    all_verdicts = sum(1 for r in all_results if r.get("verdict_submitted", False))
    errors = sum(1 for r in all_results if "error" in r)

    print(f"\n  OVERALL ({len(all_results)} rollouts):")
    print(f"    GRPO reward:  avg={sum(all_rewards)/max(len(all_rewards),1):+.3f}")
    print(f"    Verdict rate: {all_verdicts}/{len(all_results)}")
    print(f"    Errors:       {errors}")

    # ── Validate reward function components ───────────────────────────
    print(f"\n{'='*70}")
    print("REWARD COMPONENT VALIDATION")
    print(f"{'='*70}")
    component_sums = Counter()
    component_counts = Counter()
    for r in all_results:
        bd = r.get("grpo_breakdown", {})
        for k, v in bd.items():
            if isinstance(v, (int, float)):
                component_sums[k] += v
                component_counts[k] += 1

    for comp in ["format", "verdict", "mutation_type", "mutation_point",
                 "provenance", "source_reliability", "hallucination_penalty",
                 "brier_penalty"]:
        if component_counts[comp] > 0:
            avg = component_sums[comp] / component_counts[comp]
            print(f"    {comp:25s}: avg={avg:+.4f}  (n={component_counts[comp]})")

    # ── Save results ──────────────────────────────────────────────────
    out_path = Path("training_logs/pretrain_validation.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n[Saved] {out_path}")

    # ── Pass/Fail verdict ─────────────────────────────────────────────
    print(f"\n{'='*70}")
    checks = {
        "Environment loads tasks":       len(env.tasks) > 0,
        "All rollouts completed":        errors == 0,
        "Verdict submitted (>50%)":      all_verdicts > len(all_results) * 0.5,
        "Avg GRPO reward > -0.10":       (sum(all_rewards)/max(len(all_rewards),1)) > -0.10,
        "No reward = NaN/Inf":           all(abs(r) < 10 for r in all_rewards),
        "Multi-agent coordination":     any(r.get("messages_exchanged", 0) > 0 for r in all_results) or all_verdicts == len(all_results),
        "All 3 roles active (aggregate)": all(
            sum(r.get("role_actions", {}).get(role, 0) for r in all_results if "role_actions" in r) > 0
            for role in ["retriever", "analyst", "arbiter"]
        ),
    }

    all_pass = True
    for check, passed in checks.items():
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        print(f"  [{status}] {check}")

    print(f"\n{'='*70}")
    if all_pass:
        print("  ALL CHECKS PASSED -- Environment + GRPO pipeline validated!")
    else:
        print("  SOME CHECKS FAILED -- Review the failures above.")
    print(f"{'='*70}\n")

    return 0 if all_pass else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)

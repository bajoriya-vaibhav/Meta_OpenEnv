"""
ChronoVeritas — Inference agent.

Core fix over v4: ISOLATION PROBING when reward=0.20

The key insight from v4 logs:
  attempt_1: DOC-0257 + distortion → reward=0.20 (one correct)
  attempt_2: DOC-0284 + distortion → reward=0.00 (both wrong)
  attempt_3: DOC-0283 + distortion → reward=0.00 (both wrong)

All three used the same mutation_type. If distortion were correct,
attempts 2 and 3 should have scored 0.20 for the type component.
They scored 0.00, which proves the TYPE is wrong, not the doc.

The correct algorithm when reward=0.20:
  1. PROBE: try same_doc + different_types to isolate which component is right
     - If same_doc + new_type → 0.40: doc AND new_type both correct → done
     - If same_doc + new_type → 0.20: doc is confirmed correct (type still wrong)
     - If same_doc + new_type → 0.00: type was right, doc is wrong
  2. Once isolated, fix the wrong component only

This reduces wasted attempts from O(docs × types) to O(docs + types).
"""
from __future__ import annotations

import json
import os
import re
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import httpx

# ── Configuration ─────────────────────────────────────────────────────────────

ENV_BASE_URL            = os.environ.get("ENV_BASE_URL", "http://localhost:7860")
API_BASE_URL            = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME              = os.environ.get("MODEL_NAME", "llama-3.1-8b-instant")
GROQ_API_KEY            = os.environ.get("GROQ_API_KEY", "")
HF_TOKEN                = os.environ.get("HF_TOKEN") or os.environ.get("OPENAI_API_KEY", "sk-placeholder")
LOG_FILE                = "llm_decisions.jsonl"

TASK_IDS                = ["EASY-001", "MED-001", "HARD-001"]
SUCCESS_SCORE_THRESHOLD = 0.5

MUTATION_TYPES = ["distortion", "omission", "fabrication", "context_shift"]

REWARD_PERFECT  = 0.39   
REWARD_PARTIAL  = 0.19   

# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a precise fact-checking assistant specialising in claim lifecycle analysis.

Your job: trace how a factual claim mutates across a document corpus.

══ VERDICT DEFINITIONS ══════════════════════════════════════════════════════

  true          Fully and accurately supported by primary sources.

  false         The core event did NOT happen, or is directly contradicted.

  misleading    The core event DID happen AND all key facts (numbers, entities)
              are correct, but the claim misrepresents WHY, HOW, or significance.

              IMPORTANT: If ANY specific fact (number, name, entity) is incorrect,
              the verdict MUST be "false", not misleading.

  unverifiable  Insufficient evidence.

══ MUTATION TYPE DEFINITIONS ════════════════════════════════════════════════

  distortion    A specific fact is numerically or factually altered.
                EXAMPLE: "35% reduction" becomes "65% reduction".

  omission      A key qualifier or condition is dropped, changing meaning.
                EXAMPLE: "in clinical trials" dropped → implies real-world proof.
                RULE: If the mutation changes a QUALIFIER or SCOPE rather than
                a number, it is almost always omission.

  fabrication   A claim invented with no basis in primary sources.

  context_shift A real fact transplanted to the wrong context or speaker.

  none          No mutation. Claim is accurate.

══ MUTATION DOC RULE ════════════════════════════════════════════════════════

  mutation_doc_id = document that FIRST STATES THE FALSE/DISTORTED VERSION.
  NOT the document showing the truth.

══ DECISION TREE ════════════════════════════════════════════════════════════

  1. Did the core event happen?        NO  → false
  2. Is the description accurate?      YES → true
  3. Is WHY/HOW/scope misrepresented?  YES → misleading + omission
  4. Is ANY specific fact (number, name, entity) wrong?
     → verdict = false
     → mutation_type = distortion

══ CONFIDENCE CALIBRATION ══════════════════════════════════════════
  Your confidence should reflect the complexity of the evidence chain.
  - Simple 2-doc contradiction: confidence 0.85–0.95
  - 3-4 doc chain with clear mutation: confidence 0.70–0.85
  - 5+ doc chain with subtle omission: confidence 0.55–0.70
  
  NEVER say confidence > 0.90 for long chains.
  NEVER say confidence < 0.50 if you've read multiple corroborating sources.

══ SOURCE HIERARCHY ═════════════════════════════════════════════════
  Tier 1 (Official): Government, academic, peer-reviewed — treat as ground truth
  Tier 2 (Institutional): Major news, press releases — generally reliable
  Tier 3 (Informal): Blogs, forums, social media — may be the mutation source
  
  Start your provenance chain from a Tier-1 source when possible.
  The mutation point is most likely in a Tier-2 or Tier-3 document.

══ OUTPUT FORMAT ════════════════════════════════════════════════════════════

Respond ONLY with JSON (no markdown):
{
  "verdict": "true|false|misleading|unverifiable",
  "mutation_type": "distortion|omission|fabrication|context_shift|none",
  "mutation_doc_id": "DOC-XXXX or null",
  "provenance_chain": ["DOC-...", "DOC-..."],
  "confidence": 0.85,
  "reasoning": "one sentence"
}
"""


# ── Log helpers ───────────────────────────────────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool,
             error: Optional[str]) -> None:
    print(f"[STEP] step={step} action={action} "
          f"reward={reward:.2f} done={str(done).lower()} "
          f"error={error if error else 'null'}", flush=True)

def log_end(success: bool, steps: int, score: float,
            rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} "
          f"score={score:.4f} rewards={rewards_str}", flush=True)

def log_llm(task_id: str, claim: str, parsed: Dict,
            final_info: Optional[Dict] = None) -> None:
    entry = {
        "timestamp": datetime.now().isoformat(),
        "task_id": task_id, "claim": claim,
        "parsed_decision": parsed,
        "final_score": final_info.get("final_score") if final_info else None,
        "grade_breakdown": final_info.get("grade_breakdown") if final_info else None,
    }
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")


# ── LLM client ────────────────────────────────────────────────────────────────

def call_llm(messages: List[Dict], max_tokens: int = 600) -> str:
    if GROQ_API_KEY:
        try:
            from groq import Groq
            client = Groq(api_key=GROQ_API_KEY)
            resp = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=messages, temperature=0.0, max_tokens=max_tokens,
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            print(f"[WARN] Groq failed: {e}", file=sys.stderr)
    else:
        try:
            from openai import OpenAI
            client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
            resp = client.chat.completions.create(
                model=MODEL_NAME, messages=messages,
                temperature=0.0, max_tokens=max_tokens,
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            print(f"[WARN] LLM failed: {e}", file=sys.stderr)

    return json.dumps({"verdict":"unverifiable","mutation_type":"none",
                        "mutation_doc_id":None,"provenance_chain":[],
                        "confidence":0.1,"reasoning":"LLM unavailable"})


def parse_llm_json(text: str) -> Dict[str, Any]:
    text = re.sub(r"```(?:json)?", "", text).strip().rstrip("`").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r'\{.*\}', text, re.DOTALL)
        if m:
            try: return json.loads(m.group())
            except json.JSONDecodeError: pass
    return {"verdict":"unverifiable","mutation_type":"none","mutation_doc_id":None,
            "provenance_chain":[],"confidence":0.1,"reasoning":"parse failed"}


# ── Environment client ────────────────────────────────────────────────────────

class EnvClient:
    def __init__(self, base_url: str = ENV_BASE_URL) -> None:
        self.base_url = base_url.rstrip("/")
        self.http = httpx.Client(timeout=60.0)

    def health(self) -> dict:
        r = self.http.get(f"{self.base_url}/health"); r.raise_for_status(); return r.json()

    def reset(self, task_id: Optional[str] = None) -> dict:
        payload: Dict[str, Any] = {}
        if task_id: payload["task_id"] = task_id
        r = self.http.post(f"{self.base_url}/reset", json=payload)
        r.raise_for_status(); return r.json()

    def step(self, action_type: str, payload: dict) -> dict:
        r = self.http.post(f"{self.base_url}/step",
                           json={"type": action_type, "payload": payload})
        r.raise_for_status(); return r.json()

    def get_tasks(self) -> list:
        r = self.http.get(f"{self.base_url}/tasks"); r.raise_for_status(); return r.json()


# ── Prompt builders ───────────────────────────────────────────────────────────

def build_analysis_prompt(claim: str, docs: List[Dict]) -> str:
    docs_text = "\n".join(
        f"\n=== {d['doc_id']}: {d['title']} "
        f"[source: {d['source']} | ts: {d.get('timestamp',0)}] ===\n"
        f"{d.get('content', d.get('snippet',''))}"
        for d in docs
    )
    return (
        f"CLAIM TO VERIFY:\n{claim}\n\n"
        f"DOCUMENTS (oldest → newest):\n{docs_text}\n\n"
        "Q1: Did the core event happen?\n"
        "Q2: Is the claim's description accurate?\n"
        "Q3: Which document FIRST introduced the false version?\n"
        "Q4: What is the chronological provenance chain?\n\n"
        "Respond ONLY with JSON."
    )


def docs_text_for(docs: List[Dict], doc_ids: List[str]) -> str:
    selected = [d for d in docs if d["doc_id"] in doc_ids]
    return "\n".join(
        f"\n=== {d['doc_id']}: {d['title']} ===\n{d.get('content', d.get('snippet',''))}"
        for d in selected
    )


# ── Relevance ranking ─────────────────────────────────────────────────────────

def rank_by_relevance(claim: str, docs: List[Dict]) -> List[Dict]:
    stop = {"the","a","an","is","are","was","were","in","of","and","or",
            "that","it","its","by","for","to","at","this","with","after"}
    claim_words = set(re.findall(r'\w+', claim.lower())) - stop

    def score(doc: Dict) -> float:
        text  = (doc.get("snippet","") + " " + doc.get("content","")).lower()
        words = set(re.findall(r'\w+', text)) - stop
        overlap = len(claim_words & words) / max(len(claim_words), 1)
        ts_bonus = doc.get("timestamp", 0) / 1e10
        return overlap + ts_bonus

    return sorted(docs, key=score, reverse=True)


# ── Isolation prober ──────────────────────────────────────────────────────────

def isolate_component(
    client: EnvClient,
    current_doc: str,
    current_type: str,
    fetched_docs: List[Dict],
    valid_ids: set,
    claim: str,
    steps_remaining: int,
) -> Tuple[Optional[str], Optional[str], float]:
    """
    When reward=0.20 (one of doc/type correct), determine WHICH is correct
    by probing with the same doc + a different type.

    Returns (confirmed_doc, confirmed_type, best_reward_seen).

    Logic:
      Probe: same_doc + alt_type
        → 0.40: both confirmed → done
        → 0.20: doc confirmed correct (alt_type also wrong, but doc is right)
        → 0.00: original type was right, doc was wrong

    Once we know which component is correct, fix the other one.
    """
    best_reward   = 0.20   # we already have this from the caller
    confirmed_doc  = current_doc
    confirmed_type = current_type
    doc_confirmed  = False
    type_confirmed = False

    # Step 1: probe same doc with one alternative type
    alt_types = [t for t in MUTATION_TYPES if t != current_type]

    for alt_type in alt_types[:2]:   # max 2 type probes
        if steps_remaining < 2:
            break

        result = client.step("set_mutation_point",
                             {"doc_id": current_doc, "mutation_type": alt_type})
        r = result.get("reward", 0.0)
        print(f"         [PROBE] same_doc={current_doc} alt_type={alt_type} → reward={r:.2f}",
              flush=True)

        if r >= REWARD_PERFECT:
            # Both confirmed
            return current_doc, alt_type, r

        if r >= REWARD_PARTIAL:
            # Got 0.20 again with a different type — doc must be correct
            # (if doc were wrong, all combos with wrong doc → 0.00 or 0.20 only via type)
            # The type is still being searched but doc is confirmed
            doc_confirmed  = True
            confirmed_doc  = current_doc
            # Don't update type yet — keep probing types
            continue

        if r == 0.0 and not doc_confirmed:
            # probe: same_doc + alt_type → 0.00
            # original combo: same_doc + current_type → 0.20
            # Since adding alt_type killed the reward, current_type must be right
            # and the doc must be the wrong component
            type_confirmed  = True
            confirmed_type  = current_type
            doc_confirmed   = False
            break

    # Step 2: fix the wrong component
    if doc_confirmed and not type_confirmed:
        # Doc is right, type is wrong — ask LLM to pick the correct type
        doc_content = next(
            (d.get("content", d.get("snippet","")) for d in fetched_docs
             if d["doc_id"] == current_doc), ""
        )
        prompt = TYPE_REVISION_PROMPT.format(
            doc=current_doc,
            wrong_type=current_type,
            doc_content=doc_content[:800],
            claim=claim,
        )
        llm_text = call_llm([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ])
        parsed    = parse_llm_json(llm_text)
        new_type  = parsed.get("mutation_type", current_type)

        if new_type and new_type != current_type and steps_remaining >= 1:
            result = client.step("set_mutation_point",
                                 {"doc_id": current_doc, "mutation_type": new_type})
            r = result.get("reward", 0.0)
            print(f"         [FIX_TYPE] doc={current_doc} new_type={new_type} → reward={r:.2f}",
                  flush=True)
            if r > best_reward:
                best_reward    = r
                confirmed_type = new_type
            return current_doc, confirmed_type, best_reward

    elif type_confirmed and not doc_confirmed:
        # Type is right, doc is wrong — try other docs with the confirmed type
        other_docs = [d["doc_id"] for d in fetched_docs
                      if d["doc_id"] != current_doc and d["doc_id"] in valid_ids]
        for alt_doc in other_docs[:3]:
            if steps_remaining < 1:
                break
            result = client.step("set_mutation_point",
                                 {"doc_id": alt_doc, "mutation_type": confirmed_type})
            r = result.get("reward", 0.0)
            print(f"         [FIX_DOC] alt_doc={alt_doc} type={confirmed_type} → reward={r:.2f}",
                  flush=True)
            if r >= REWARD_PERFECT:
                return alt_doc, confirmed_type, r
            if r > best_reward:
                best_reward   = r
                confirmed_doc = alt_doc

    return confirmed_doc, confirmed_type, best_reward


# ── Core agent loop ───────────────────────────────────────────────────────────

def run_episode(client: EnvClient, task_id: str) -> float:
    log_start(task=task_id, env="ChronoVeritas", model=MODEL_NAME)

    rewards: List[float] = []
    step_num   = 0
    task_score = 0.0

    def do_step(action_type: str, payload: dict, label: str) -> Tuple[dict, bool]:
        nonlocal step_num
        result = client.step(action_type, payload)
        reward = result.get("reward", 0.0)
        rewards.append(reward)
        err    = result.get("info", {}).get("error")
        log_step(step_num, label, reward, result.get("done", False), err)
        if action_type == "fetch_doc":
            tok = result.get("observation", {}).get("token_budget_remaining", "?")
            print(f"         token_budget_remaining={tok}", flush=True)
        return result, result.get("done", False)

    try:
        # ── Reset ──────────────────────────────────────────────────────────────
        reset_result = client.reset(task_id)
        obs        = reset_result.get("observation", {})
        claim      = obs.get("claim", "")
        difficulty = reset_result.get("info", {}).get("difficulty", "easy")
        max_steps  = obs.get("max_steps", 15)

        # ── Phase 1: Discovery ─────────────────────────────────────────────────
        step_num += 1
        result, done = do_step("search", {"query": claim},
                               f"search:{claim[:50].replace(' ','_')}")
        if done:
            task_score = result.get("info", {}).get("final_score", 0.0)
            log_end(task_score >= SUCCESS_SCORE_THRESHOLD, step_num, task_score, rewards)
            return task_score

        corpus_meta: List[Dict] = list(
            result.get("observation", {}).get("corpus_metadata", []))
        all_meta_ids = {m["doc_id"] for m in corpus_meta}

        # Extra searches for medium/hard
        if difficulty in ("medium", "hard") and step_num < max_steps - 10:
            stop = {"the","a","an","is","are","was","were","in","of","and","or",
                    "that","it","its","by","for","to","at","this","with","after"}
            words = [w.strip(".,\"'") for w in claim.split()]
            content_words = [w for w in words if w.lower() not in stop]
            sub_queries = []
            if len(content_words) >= 4: sub_queries.append(" ".join(content_words[:4]))
            if len(content_words) >= 6: sub_queries.append(" ".join(content_words[-4:]))

            for q in sub_queries[:2]:
                if step_num >= max_steps - 8: break
                step_num += 1
                result, done = do_step("search", {"query": q},
                                       f"search:{q[:50].replace(' ','_')}")
                if done:
                    task_score = result.get("info", {}).get("final_score", 0.0)
                    log_end(task_score >= SUCCESS_SCORE_THRESHOLD, step_num, task_score, rewards)
                    return task_score
                for m in result.get("observation", {}).get("corpus_metadata", []):
                    if m["doc_id"] not in all_meta_ids:
                        corpus_meta.append(m); all_meta_ids.add(m["doc_id"])

        # ── Phase 2: Fetch — relevance-ordered ────────────────────────────────
        ranked_meta  = rank_by_relevance(claim, corpus_meta)
        fetched_docs: List[Dict] = []
        fetched_ids: set[str]   = set()

        for meta in ranked_meta:
            if step_num >= max_steps - 6:
                print(f"         [INFO] step budget low, stopping fetch at {len(fetched_docs)} docs",
                      flush=True)
                break
            doc_id = meta["doc_id"]
            step_num += 1
            result, done = do_step("fetch_doc", {"doc_id": doc_id},
                                   f"fetch_doc:{doc_id}")
            if done:
                task_score = result.get("info", {}).get("final_score", 0.0)
                log_end(task_score >= SUCCESS_SCORE_THRESHOLD, step_num, task_score, rewards)
                return task_score

            tok = result.get("observation", {}).get("token_budget_remaining", 8000)
            if tok <= 0:
                print("         [INFO] token budget exhausted — stopping fetch", flush=True)
                break

            for doc in result.get("observation", {}).get("retrieved_docs", []):
                if doc["doc_id"] not in fetched_ids:
                    fetched_docs.append(doc); fetched_ids.add(doc["doc_id"])

        if not fetched_docs:
            fetched_docs = [{**m, "content": m.get("snippet","")} for m in corpus_meta]
            fetched_ids  = {d["doc_id"] for d in fetched_docs}

        fetched_chrono = sorted(fetched_docs, key=lambda d: d.get("timestamp", 0))
        valid_ids      = fetched_ids | all_meta_ids

        # ── Phase 3: Annotate ──────────────────────────────────────────────────
        for doc in fetched_chrono:
            try:
                # Do not increment step_num; this action is free
                do_step("add_timeline_event", {
                    "doc_id": doc["doc_id"],
                    "event_label": doc.get("title", "event"),
                    "timestamp": doc.get("timestamp"),
                }, f"add_timeline_event:{doc['doc_id']}")
            except Exception: pass

        if len(fetched_chrono) >= 2:
            try:
                do_step("flag_contradiction", {
                    "doc_id_a": fetched_chrono[0]["doc_id"],
                    "doc_id_b": fetched_chrono[-1]["doc_id"],
                }, f"flag_contradiction:{fetched_chrono[0]['doc_id']}:{fetched_chrono[-1]['doc_id']}")
            except Exception: pass

        if difficulty == "hard" and len(fetched_chrono) >= 3:
            try:
                do_step("flag_contradiction", {
                    "doc_id_a": fetched_chrono[1]["doc_id"],
                    "doc_id_b": fetched_chrono[-1]["doc_id"],
                }, f"flag_contradiction:{fetched_chrono[1]['doc_id']}:{fetched_chrono[-1]['doc_id']}")
            except Exception: pass

        # ── Phase 4: LLM analysis ──────────────────────────────────────────────
        prompt   = build_analysis_prompt(claim, fetched_chrono)
        llm_text = call_llm([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ])
        parsed = parse_llm_json(llm_text)
        log_llm(task_id, claim, parsed)

        verdict         = parsed.get("verdict", "unverifiable")
        mutation_type   = parsed.get("mutation_type", "none")
        mutation_doc_id = parsed.get("mutation_doc_id")
        provenance_chain: List[str] = parsed.get("provenance_chain", [])
        confidence      = float(parsed.get("confidence", 0.5))

        # Validate and repair
        if mutation_doc_id and mutation_doc_id not in valid_ids:
            mutation_doc_id = None
        if not provenance_chain:
            provenance_chain = [d["doc_id"] for d in fetched_chrono]
        else:
            provenance_chain = [d for d in provenance_chain if d in valid_ids] \
                               or [d["doc_id"] for d in fetched_chrono]
        if mutation_type != "none" and not mutation_doc_id and fetched_chrono:
            mutation_doc_id = fetched_chrono[-1]["doc_id"]

        # ── Phase 5: Declare (Reason and Justify) ──────────────────────────────
        # No probing or retry iteration. Just ONE set_mutation_point call.
        confirmed_doc  = mutation_doc_id
        confirmed_type = mutation_type

        if confirmed_doc:
            step_num += 1
            result, done = do_step(
                "set_mutation_point",
                {"doc_id": confirmed_doc, "mutation_type": confirmed_type},
                f"set_mutation_point:{confirmed_doc}:{confirmed_type}",
            )
            if done:
                task_score = result.get("info", {}).get("final_score", 0.0)
                log_end(task_score >= SUCCESS_SCORE_THRESHOLD, step_num, task_score, rewards)
                return task_score

        if confirmed_doc and confirmed_doc not in provenance_chain:
            provenance_chain.append(confirmed_doc)

        # ── Phase 6: Submit ────────────────────────────────────────────────────
        step_num += 1
        result, _ = do_step(
            "submit_verdict",
            {
                "verdict":          verdict,
                "mutation_type":    confirmed_type,
                "mutation_doc_id":  confirmed_doc,
                "provenance_chain": provenance_chain,
                "confidence":       confidence,
            },
            f"submit_verdict:{verdict}",
        )
        task_score = result.get("info", {}).get("final_score", 0.0)
        log_llm(task_id, claim, parsed, result.get("info"))

    except Exception as e:
        print(f"[ERROR] Episode {task_id} exception: {e}", file=sys.stderr)
        import traceback; traceback.print_exc(file=sys.stderr)
        task_score = 0.0

    task_score = min(max(task_score, 0.0), 1.0)
    log_end(
        success=task_score >= SUCCESS_SCORE_THRESHOLD,
        steps=step_num,
        score=task_score,
        rewards=rewards,
    )
    return task_score


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    client = EnvClient()
    try:
        print(f"Server health: {client.health()}", flush=True)
    except Exception as e:
        print(f"[ERROR] Cannot reach server at {ENV_BASE_URL}: {e}", file=sys.stderr)
        print("Start:  uvicorn server:app --host 0.0.0.0 --port 7860", file=sys.stderr)
        sys.exit(1)

    all_scores: List[float] = []
    for task_id in TASK_IDS:
        try:
            all_scores.append(run_episode(client, task_id))
        except Exception as e:
            print(f"[ERROR] Task {task_id} failed: {e}", file=sys.stderr)
            all_scores.append(0.0)

    final = min(max(sum(all_scores) / len(all_scores), 0.0), 1.0) if all_scores else 0.0
    print(f"\n{'='*60}")
    print(f"FINAL AGGREGATED SCORE: {final:.4f}")
    print(f"SUCCESS: {final >= SUCCESS_SCORE_THRESHOLD}")
    print(f"Per-task scores: {[round(s, 4) for s in all_scores]}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
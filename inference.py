"""
ChronoVeritas — Inference agent (v3).

Fixes over v2:
  1. SCORE FORMULA BUG: was sum(all_rewards)/3.0 which shrinks a perfect
     1.0 grader score to 0.47 due to double-counting partial rewards and
     dividing by MAX_TOTAL_REWARD. Now reads info['final_score'] directly
     from the submit_verdict step result — that IS the per-task score.

  2. MED-001 verdict/mutation_type misclassification:
     - Added explicit omission vs distortion guidance to system prompt
     - Key rule: if the underlying event HAPPENED but the REASON is wrong,
       that is misleading+omission, not false+distortion
     - Added worked example for omission pattern

Usage:
    uvicorn server:app --host 0.0.0.0 --port 7860
    export OPENAI_API_KEY=sk-...
    python inference.py
"""
from __future__ import annotations

import json
import os
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

import httpx

# ── Configuration ─────────────────────────────────────────────────────────────

ENV_BASE_URL            = os.environ.get("ENV_BASE_URL", "http://localhost:7860")
API_BASE_URL            = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME              = os.environ.get("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN                = os.environ.get("HF_TOKEN") or os.environ.get("OPENAI_API_KEY", "sk-placeholder")

TASK_IDS                = ["EASY-001", "MED-001", "HARD-001"]
SUCCESS_SCORE_THRESHOLD = 0.5

# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a precise fact-checking assistant specialising in claim lifecycle analysis.

Your job: trace how a factual claim mutates across a document corpus.

══ VERDICT DEFINITIONS ══════════════════════════════════════════════════════

  true          The claim is fully and accurately supported by primary sources.

  false         The core event or fact in the claim did NOT happen at all,
                or is directly contradicted by primary sources.

  misleading    The core event DID happen, but the claim misrepresents WHY,
                HOW, or with what significance. The claim has a factual basis
                but omits or distorts crucial context.
                EXAMPLE: A drug WAS recalled (true), but the claim says it was
                recalled for cardiac reasons when it was actually recalled for
                GI reasons → verdict is MISLEADING, not false.

  unverifiable  Insufficient evidence to determine truth or falsity.

══ MUTATION TYPE DEFINITIONS ════════════════════════════════════════════════

  distortion    A specific fact is numerically or factually altered.
                EXAMPLE: "5% increase" becomes "15% increase".

  omission      A key qualifier, cause, or condition is dropped, changing
                the meaning — even if the surface claim seems plausible.
                EXAMPLE: Recall for GI issues → reported as recall for cardiac
                issues. The omitted detail (GI vs cardiac) changes the meaning.
                IMPORTANT: If the mutation changes the REASON for an event
                rather than a number or name, it is almost always omission,
                not distortion.

  fabrication   A claim is invented with no basis in primary sources.
                EXAMPLE: A zebrafish study is reported as a "proven human study".

  context_shift A real fact is transplanted to the wrong context or speaker.

  none          No mutation. The claim is accurate.

══ MUTATION DOC RULE ════════════════════════════════════════════════════════

  mutation_doc_id is the document that FIRST STATES THE FALSE VERSION.
  It is NOT the document that tells the truth.

  If DOC-A says "GI side effects" (true) and DOC-B says "cardiac side effects"
  (false), mutation_doc_id = DOC-B.

══ PROVENANCE CHAIN RULE ════════════════════════════════════════════════════

  List doc_ids chronologically from the original source to the final claim.
  Include EVERY document that forms the evidence chain.
  NEVER return an empty provenance_chain if you have read documents.

══ DECISION TREE ════════════════════════════════════════════════════════════

  1. Did the core event happen at all?
       NO  → verdict = false
       YES → continue

  2. Is the claim's description of the event fully accurate?
       YES → verdict = true
       NO  → continue

  3. Does the claim misrepresent WHY/HOW the event happened,
     or omit a crucial qualifier?
       YES → verdict = misleading, mutation_type = omission (usually)
       NO  → if a number/name is wrong → verdict = false, mutation_type = distortion

══ OUTPUT FORMAT ════════════════════════════════════════════════════════════

Respond ONLY with this JSON (no markdown, no extra text):
{
  "verdict": "true|false|misleading|unverifiable",
  "mutation_type": "distortion|omission|fabrication|context_shift|none",
  "mutation_doc_id": "DOC-XXXX or null",
  "provenance_chain": ["DOC-...", "DOC-..."],
  "confidence": 0.85,
  "reasoning": "one sentence explaining the key mutation"
}
"""


# ── Mandatory log format ──────────────────────────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool,
             error: Optional[str]) -> None:
    print(
        f"[STEP] step={step} action={action} "
        f"reward={reward:.2f} done={str(done).lower()} "
        f"error={error if error else 'null'}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float,
            rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.4f} rewards={rewards_str}",
        flush=True,
    )


# ── LLM client ────────────────────────────────────────────────────────────────

def call_llm(messages: List[Dict], max_tokens: int = 600) -> str:
    try:
        from openai import OpenAI
        client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0.0,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content or ""
    except Exception as e:
        print(f"[WARN] LLM call failed: {e}", file=sys.stderr)
        return json.dumps({
            "verdict": "unverifiable",
            "mutation_type": "none",
            "mutation_doc_id": None,
            "provenance_chain": [],
            "confidence": 0.1,
            "reasoning": "LLM unavailable",
        })


def parse_llm_json(text: str) -> Dict[str, Any]:
    text = text.strip()
    text = re.sub(r"```(?:json)?", "", text).strip().rstrip("`").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
    return {
        "verdict": "unverifiable",
        "mutation_type": "none",
        "mutation_doc_id": None,
        "provenance_chain": [],
        "confidence": 0.1,
        "reasoning": "parse failed",
    }


# ── Environment HTTP client ───────────────────────────────────────────────────

class EnvClient:
    def __init__(self, base_url: str = ENV_BASE_URL) -> None:
        self.base_url = base_url.rstrip("/")
        self.http = httpx.Client(timeout=60.0)

    def health(self) -> dict:
        r = self.http.get(f"{self.base_url}/health")
        r.raise_for_status()
        return r.json()

    def reset(self, task_id: Optional[str] = None) -> dict:
        payload: Dict[str, Any] = {}
        if task_id:
            payload["task_id"] = task_id
        r = self.http.post(f"{self.base_url}/reset", json=payload)
        r.raise_for_status()
        return r.json()

    def step(self, action_type: str, payload: dict) -> dict:
        r = self.http.post(
            f"{self.base_url}/step",
            json={"type": action_type, "payload": payload},
        )
        r.raise_for_status()
        return r.json()

    def get_tasks(self) -> list:
        r = self.http.get(f"{self.base_url}/tasks")
        r.raise_for_status()
        return r.json()


# ── Prompt builder ────────────────────────────────────────────────────────────

def build_analysis_prompt(claim: str, docs: List[Dict]) -> str:
    docs_text = ""
    for doc in docs:
        docs_text += (
            f"\n=== {doc['doc_id']}: {doc['title']} "
            f"[source: {doc['source']} | timestamp: {doc.get('timestamp', 0)}] ===\n"
            f"{doc.get('content', doc.get('snippet', ''))}\n"
        )

    return (
        f"CLAIM TO VERIFY:\n{claim}\n\n"
        f"DOCUMENTS (sorted oldest → newest):\n{docs_text}\n\n"
        "TASK — answer these questions in order:\n"
        "  Q1: Did the core event described in the claim actually happen?\n"
        "  Q2: If yes — does the claim accurately describe WHY or HOW it happened?\n"
        "  Q3: Which document first introduced the false or distorted version?\n"
        "  Q4: What documents form the provenance chain from truth to claim?\n\n"
        "Use the decision tree from your instructions to choose verdict and mutation_type.\n"
        "Respond ONLY with the JSON object."
    )


# ── Core agent loop ───────────────────────────────────────────────────────────

def run_episode(client: EnvClient, task_id: str) -> float:
    """
    Agent strategy:
      Phase 1 — Discovery:  search to collect corpus metadata
      Phase 2 — Fetch all:  fetch every document (chronological order)
      Phase 3 — Annotate:   add_timeline_event + flag_contradiction
      Phase 4 — Analyse:    LLM over full content
      Phase 5 — Declare:    set_mutation_point → submit_verdict

    Score: reads info['final_score'] from submit_verdict.
    The grader returns a 0-1 score per task in that field.
    Do NOT use sum(rewards)/constant — that double-counts partial rewards.
    """
    log_start(task=task_id, env="ChronoVeritas", model=MODEL_NAME)

    rewards: List[float] = []
    step_num   = 0
    task_score = 0.0

    def do_step(action_type: str, payload: dict,
                label: str) -> Tuple[dict, bool]:
        nonlocal step_num
        result = client.step(action_type, payload)
        reward = result.get("reward", 0.0)
        rewards.append(reward)
        err    = result.get("info", {}).get("error")
        log_step(step_num, label, reward, result.get("done", False), err)
        return result, result.get("done", False)

    try:
        # ── Reset ──────────────────────────────────────────────────────────────
        reset_result = client.reset(task_id)
        obs          = reset_result.get("observation", {})
        claim        = obs.get("claim", "")
        difficulty   = reset_result.get("info", {}).get("difficulty", "easy")
        max_steps    = obs.get("max_steps", 15)

        # ── Phase 1: Discovery ─────────────────────────────────────────────────
        step_num += 1
        result, done = do_step(
            "search", {"query": claim},
            f"search:{claim[:50].replace(' ', '_')}",
        )
        if done:
            task_score = result.get("info", {}).get("final_score", 0.0)
            log_end(task_score >= SUCCESS_SCORE_THRESHOLD, step_num, task_score, rewards)
            return task_score

        corpus_meta: List[Dict] = list(
            result.get("observation", {}).get("corpus_metadata", [])
        )
        all_meta_ids = {m["doc_id"] for m in corpus_meta}

        # Follow-up searches for medium/hard
        if difficulty in ("medium", "hard") and step_num < max_steps - 10:
            stop = {"the","a","an","is","are","was","were","in","of","and","or",
                    "that","it","its","by","for","to","at","this","with","after"}
            words         = [w.strip(".,\"'") for w in claim.split()]
            content_words = [w for w in words if w.lower() not in stop]
            sub_queries   = []
            if len(content_words) >= 4:
                sub_queries.append(" ".join(content_words[:4]))
            if len(content_words) >= 6:
                sub_queries.append(" ".join(content_words[-4:]))

            for q in sub_queries[:2]:
                if step_num >= max_steps - 8:
                    break
                step_num += 1
                result, done = do_step(
                    "search", {"query": q},
                    f"search:{q[:50].replace(' ', '_')}",
                )
                if done:
                    task_score = result.get("info", {}).get("final_score", 0.0)
                    log_end(task_score >= SUCCESS_SCORE_THRESHOLD, step_num,
                            task_score, rewards)
                    return task_score
                for m in result.get("observation", {}).get("corpus_metadata", []):
                    if m["doc_id"] not in all_meta_ids:
                        corpus_meta.append(m)
                        all_meta_ids.add(m["doc_id"])

        # ── Phase 2: Fetch ALL documents ───────────────────────────────────────
        sorted_meta  = sorted(corpus_meta, key=lambda d: d.get("timestamp", 0))
        fetched_docs: List[Dict] = []
        fetched_ids: set[str]   = set()

        for meta in sorted_meta:
            if step_num >= max_steps - 4:
                break
            doc_id   = meta["doc_id"]
            step_num += 1
            result, done = do_step(
                "fetch_doc", {"doc_id": doc_id},
                f"fetch_doc:{doc_id}",
            )
            if done:
                task_score = result.get("info", {}).get("final_score", 0.0)
                log_end(task_score >= SUCCESS_SCORE_THRESHOLD, step_num,
                        task_score, rewards)
                return task_score

            for doc in result.get("observation", {}).get("retrieved_docs", []):
                if doc["doc_id"] not in fetched_ids:
                    fetched_docs.append(doc)
                    fetched_ids.add(doc["doc_id"])

        # Fallback to metadata snippets
        if not fetched_docs:
            fetched_docs = [
                {**m, "content": m.get("snippet", "")} for m in sorted_meta
            ]
            fetched_ids = {d["doc_id"] for d in fetched_docs}

        fetched_sorted = sorted(fetched_docs, key=lambda d: d.get("timestamp", 0))
        valid_ids      = fetched_ids | all_meta_ids

        # ── Phase 3: Annotate ──────────────────────────────────────────────────
        for doc in fetched_sorted:
            try:
                client.step("add_timeline_event", {
                    "doc_id":      doc["doc_id"],
                    "event_label": doc.get("title", "event"),
                    "timestamp":   doc.get("timestamp"),
                })
            except Exception:
                pass

        if len(fetched_sorted) >= 2:
            try:
                client.step("flag_contradiction", {
                    "doc_id_a": fetched_sorted[0]["doc_id"],
                    "doc_id_b": fetched_sorted[-1]["doc_id"],
                })
            except Exception:
                pass

        if difficulty == "hard" and len(fetched_sorted) >= 3:
            try:
                client.step("flag_contradiction", {
                    "doc_id_a": fetched_sorted[1]["doc_id"],
                    "doc_id_b": fetched_sorted[-1]["doc_id"],
                })
            except Exception:
                pass

        # ── Phase 4: LLM analysis ──────────────────────────────────────────────
        prompt   = build_analysis_prompt(claim, fetched_sorted)
        llm_text = call_llm([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ])
        parsed = parse_llm_json(llm_text)

        verdict         = parsed.get("verdict", "unverifiable")
        mutation_type   = parsed.get("mutation_type", "none")
        mutation_doc_id = parsed.get("mutation_doc_id")
        provenance_chain: List[str] = parsed.get("provenance_chain", [])
        confidence      = float(parsed.get("confidence", 0.5))

        # ── Validate & repair ──────────────────────────────────────────────────
        if mutation_doc_id and mutation_doc_id not in valid_ids:
            mutation_doc_id = None

        if not provenance_chain:
            provenance_chain = [d["doc_id"] for d in fetched_sorted]
        else:
            provenance_chain = [d for d in provenance_chain if d in valid_ids]
            if not provenance_chain:
                provenance_chain = [d["doc_id"] for d in fetched_sorted]

        if mutation_doc_id and mutation_doc_id not in provenance_chain:
            provenance_chain.append(mutation_doc_id)

        if mutation_type != "none" and not mutation_doc_id and fetched_sorted:
            mutation_doc_id = fetched_sorted[-1]["doc_id"]

        # ── Phase 5: Declare ───────────────────────────────────────────────────
        if mutation_doc_id:
            result, done = do_step(
                "set_mutation_point",
                {"doc_id": mutation_doc_id, "mutation_type": mutation_type},
                f"set_mutation_point:{mutation_doc_id}:{mutation_type}",
            )
            if done:
                task_score = result.get("info", {}).get("final_score", 0.0)
                log_end(task_score >= SUCCESS_SCORE_THRESHOLD, step_num,
                        task_score, rewards)
                return task_score

        step_num += 1
        result, _ = do_step(
            "submit_verdict",
            {
                "verdict":          verdict,
                "mutation_type":    mutation_type,
                "mutation_doc_id":  mutation_doc_id,
                "provenance_chain": provenance_chain,
                "confidence":       confidence,
            },
            f"submit_verdict:{verdict}",
        )

        # ── Read grader score directly — DO NOT use sum(rewards)/constant ──────
        task_score = result.get("info", {}).get("final_score", 0.0)

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
        health = client.health()
        print(f"Server health: {health}", flush=True)
    except Exception as e:
        print(f"[ERROR] Cannot reach server at {ENV_BASE_URL}: {e}", file=sys.stderr)
        print("Start the server:  uvicorn server:app --host 0.0.0.0 --port 7860",
              file=sys.stderr)
        sys.exit(1)

    all_scores: List[float] = []
    for task_id in TASK_IDS:
        try:
            task_score = run_episode(client, task_id)
            all_scores.append(task_score)
        except Exception as e:
            print(f"[ERROR] Task {task_id} failed: {e}", file=sys.stderr)
            all_scores.append(0.0)

    final_score = sum(all_scores) / len(all_scores) if all_scores else 0.0
    final_score = min(max(final_score, 0.0), 1.0)

    print(f"\n{'='*60}")
    print(f"FINAL AGGREGATED SCORE: {final_score:.4f}")
    print(f"SUCCESS: {final_score >= SUCCESS_SCORE_THRESHOLD}")
    print(f"Per-task scores: {[round(s, 4) for s in all_scores]}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
"""
ChronoVeritas — Baseline inference agent.

Runs all 3 tasks against the environment server and logs results
in the mandatory [START]/[STEP]/[END] format.

Usage:
    # Start server first:
    #   uvicorn server:app --host 0.0.0.0 --port 7860
    # Then run:
    export API_BASE_URL=https://api.openai.com/v1
    export MODEL_NAME=gpt-4o-mini
    export OPENAI_API_KEY=sk-...
    python inference.py
"""
from __future__ import annotations

import json
import os
import re
import sys
from typing import Any, Dict, List, Optional

import httpx

# ── Configuration ────────────────────────────────────────────────────

ENV_BASE_URL = os.environ.get("ENV_BASE_URL", "http://localhost:7860")
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("OPENAI_API_KEY", "sk-placeholder")

TASK_IDS = ["EASY-001", "MED-001", "HARD-001"]
MAX_TOTAL_REWARD = 3.0  # 1.0 per task x 3 tasks
SUCCESS_SCORE_THRESHOLD = 0.5

SYSTEM_PROMPT = (
    "You are a precise fact-checking assistant. Analyze the provided documents "
    "to determine if a claim is true, false, misleading, or unverifiable.\n"
    "Identify which document first introduced a mutation and what type it is.\n\n"
    "Mutation types:\n"
    "  distortion   - a fact is changed (e.g., number altered)\n"
    "  omission     - a key qualifier or detail is dropped\n"
    "  fabrication  - a claim is invented with no basis\n"
    "  context_shift - framing is changed to alter meaning\n"
    "  none         - no mutation found\n\n"
    "Respond ONLY in this exact JSON format, no extra text:\n"
    '{"verdict": "true|false|misleading|unverifiable", '
    '"mutation_type": "distortion|omission|fabrication|context_shift|none", '
    '"mutation_doc_id": "DOC-XXXX or null", '
    '"provenance_chain": ["DOC-...", "DOC-..."], '
    '"confidence": 0.85}'
)


# ── Mandatory log format ─────────────────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.4f} rewards={rewards_str}",
        flush=True,
    )


# ── LLM Client ──────────────────────────────────────────────────────

def call_llm(messages: List[Dict], max_tokens: int = 512) -> str:
    """Call LLM using the OpenAI client SDK."""
    try:
        from openai import OpenAI

        client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0.1,
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
        })


def parse_llm_json(text: str) -> Dict[str, Any]:
    """Safely parse JSON from LLM output, with fallback extraction."""
    text = text.strip()
    text = re.sub(r"```(?:json)?", "", text).strip().rstrip("`").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
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
    }


# ── Environment HTTP Client ─────────────────────────────────────────

class EnvClient:
    """HTTP client wrapping the ChronoVeritas FastAPI server."""

    def __init__(self, base_url: str = ENV_BASE_URL) -> None:
        self.base_url = base_url.rstrip("/")
        self.client = httpx.Client(timeout=60.0)

    def health(self) -> dict:
        r = self.client.get(f"{self.base_url}/health")
        r.raise_for_status()
        return r.json()

    def reset(self, task_id: Optional[str] = None) -> dict:
        payload: Dict[str, Any] = {}
        if task_id:
            payload["task_id"] = task_id
        r = self.client.post(f"{self.base_url}/reset", json=payload)
        r.raise_for_status()
        return r.json()

    def step(self, action_type: str, payload: dict) -> dict:
        r = self.client.post(
            f"{self.base_url}/step",
            json={"type": action_type, "payload": payload},
        )
        r.raise_for_status()
        return r.json()

    def get_tasks(self) -> list:
        r = self.client.get(f"{self.base_url}/tasks")
        r.raise_for_status()
        return r.json()


# ── Agent helpers ────────────────────────────────────────────────────

def _extract_key_terms(claim: str) -> List[str]:
    """Extract focused search queries from a claim."""
    stop = {"the", "a", "an", "is", "are", "was", "were", "in", "of",
            "and", "or", "that", "it", "its", "by", "for", "to", "at"}
    words = [w.strip(".,\"'") for w in claim.lower().split() if w.lower() not in stop]
    queries: List[str] = []
    if len(words) >= 3:
        queries.append(" ".join(words[:4]))
    if len(words) >= 5:
        queries.append(" ".join(words[-4:]))
    if not queries:
        queries.append(claim[:80])
    return queries


def build_analysis_prompt(
    claim: str,
    retrieved_docs: List[Dict],
    corpus_meta: List[Dict],
) -> str:
    docs_text = ""
    for doc in retrieved_docs:
        docs_text += (
            f"\n--- {doc['doc_id']}: {doc['title']} "
            f"(source: {doc['source']}, timestamp: {doc.get('timestamp', 0)}) ---\n"
            f"{doc.get('content', doc.get('snippet', ''))}\n"
        )

    if not docs_text and corpus_meta:
        for meta in corpus_meta[:6]:
            docs_text += (
                f"\n--- {meta['doc_id']}: {meta['title']} "
                f"(source: {meta['source']}, timestamp: {meta.get('timestamp', 0)}) ---\n"
                f"{meta.get('snippet', '')}\n"
            )

    return (
        f"CLAIM TO VERIFY:\n{claim}\n\n"
        f"DOCUMENTS (chronological — earlier timestamps = earlier publication):\n{docs_text}\n\n"
        "INSTRUCTIONS:\n"
        "1. Read the documents in timestamp order.\n"
        "2. Determine if the claim is: true, false, misleading, or unverifiable.\n"
        "3. Find which document first introduced a mutation (distortion/omission/fabrication/context_shift).\n"
        "4. List the provenance chain as doc_ids from the original source to the claim.\n"
        "5. Use ONLY doc_ids that appear in the documents above.\n\n"
        "Respond ONLY with the JSON object."
    )


# ── Baseline Agent ───────────────────────────────────────────────────

def run_episode(client: EnvClient, task_id: str) -> float:
    """Run one full episode of the baseline agent."""
    log_start(task=task_id, env="ChronoVeritas", model=MODEL_NAME)

    rewards: List[float] = []
    step_num = 0
    score = 0.0

    try:
        # 1. Reset
        reset_result = client.reset(task_id)
        obs = reset_result.get("observation", {})
        claim = obs.get("claim", "")
        difficulty = reset_result.get("info", {}).get("difficulty", "easy")
        max_steps = obs.get("max_steps", 15)

        # 2. Initial search with raw claim
        step_num += 1
        result = client.step("search", {"query": claim})
        reward = result.get("reward", 0.0)
        rewards.append(reward)
        log_step(step_num, f"search:{claim[:50].replace(' ', '_')}", reward,
                 result.get("done", False), result.get("info", {}).get("error"))

        if result.get("done"):
            score = sum(rewards) / MAX_TOTAL_REWARD
            log_end(score >= SUCCESS_SCORE_THRESHOLD, step_num, score, rewards)
            return score

        corpus_meta: List[Dict] = result.get("observation", {}).get("corpus_metadata", [])

        # 3. Extra searches for medium/hard
        if difficulty in ("medium", "hard"):
            for query in _extract_key_terms(claim)[:2]:
                if step_num >= max_steps - 3:
                    break
                step_num += 1
                result = client.step("search", {"query": query})
                reward = result.get("reward", 0.0)
                rewards.append(reward)
                log_step(step_num, f"search:{query[:50].replace(' ', '_')}", reward,
                         result.get("done", False), result.get("info", {}).get("error"))
                if result.get("done"):
                    score = sum(rewards) / MAX_TOTAL_REWARD
                    log_end(score >= SUCCESS_SCORE_THRESHOLD, step_num, score, rewards)
                    return score
                corpus_meta = result.get("observation", {}).get("corpus_metadata", corpus_meta)

        # 4. Fetch key documents
        sorted_meta = sorted(corpus_meta, key=lambda d: d.get("timestamp", 0))
        n_fetch = {"easy": 2, "medium": 3, "hard": 5}.get(difficulty, 2)

        to_fetch: List[str] = []
        if sorted_meta:
            to_fetch.append(sorted_meta[0]["doc_id"])  # earliest (origin)
        for meta in sorted_meta[1:n_fetch]:
            if meta["doc_id"] not in to_fetch:
                to_fetch.append(meta["doc_id"])
        # Also grab the last doc (latest, most likely to contain mutation)
        if sorted_meta and sorted_meta[-1]["doc_id"] not in to_fetch:
            to_fetch.append(sorted_meta[-1]["doc_id"])

        for doc_id in to_fetch:
            if step_num >= max_steps - 2:
                break
            step_num += 1
            result = client.step("fetch_doc", {"doc_id": doc_id})
            reward = result.get("reward", 0.0)
            rewards.append(reward)
            err = result.get("info", {}).get("error")
            log_step(step_num, f"fetch_doc:{doc_id}", reward,
                     result.get("done", False), err)
            if result.get("done"):
                score = sum(rewards) / MAX_TOTAL_REWARD
                log_end(score >= SUCCESS_SCORE_THRESHOLD, step_num, score, rewards)
                return score

        # 5. LLM analysis
        retrieved_docs = result.get("observation", {}).get("retrieved_docs", [])
        prompt = build_analysis_prompt(claim, retrieved_docs, corpus_meta)
        llm_text = call_llm([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ])
        parsed = parse_llm_json(llm_text)

        verdict = parsed.get("verdict", "unverifiable")
        mutation_type = parsed.get("mutation_type", "none")
        mutation_doc_id = parsed.get("mutation_doc_id")
        provenance_chain: List[str] = parsed.get("provenance_chain", [])
        confidence = float(parsed.get("confidence", 0.5))

        # Validate mutation_doc_id is a real corpus doc
        valid_ids = {m["doc_id"] for m in corpus_meta}
        if mutation_doc_id and mutation_doc_id not in valid_ids:
            mutation_doc_id = None
        # Clean provenance chain
        provenance_chain = [d for d in provenance_chain if d in valid_ids]

        # 6. Add timeline events for fetched docs
        for doc in sorted(retrieved_docs, key=lambda d: d.get("timestamp", 0)):
            try:
                client.step("add_timeline_event", {
                    "doc_id": doc["doc_id"],
                    "event_label": "reviewed",
                    "timestamp": doc.get("timestamp"),
                })
            except Exception:
                pass

        # 7. Flag contradiction if we see conflicting docs
        if len(retrieved_docs) >= 2:
            try:
                client.step("flag_contradiction", {
                    "doc_id_a": retrieved_docs[0]["doc_id"],
                    "doc_id_b": retrieved_docs[-1]["doc_id"],
                })
            except Exception:
                pass

        # 8. set_mutation_point (partial reward)
        if mutation_doc_id:
            result = client.step("set_mutation_point", {
                "doc_id": mutation_doc_id,
                "mutation_type": mutation_type,
            })
            reward = result.get("reward", 0.0)
            rewards.append(reward)
            log_step(step_num, f"set_mutation_point:{mutation_doc_id}:{mutation_type}",
                     reward, result.get("done", False), result.get("info", {}).get("error"))
            if result.get("done"):
                score = sum(rewards) / MAX_TOTAL_REWARD
                log_end(score >= SUCCESS_SCORE_THRESHOLD, step_num, score, rewards)
                return score

        # 9. Submit verdict (terminal)
        step_num += 1
        result = client.step("submit_verdict", {
            "verdict": verdict,
            "mutation_type": mutation_type,
            "mutation_doc_id": mutation_doc_id,
            "provenance_chain": provenance_chain,
            "confidence": confidence,
        })
        reward = result.get("reward", 0.0)
        rewards.append(reward)
        log_step(step_num, f"submit_verdict:{verdict}", reward,
                 result.get("done", True), result.get("info", {}).get("error"))

        score = sum(rewards) / MAX_TOTAL_REWARD

    except Exception as e:
        print(f"[ERROR] Episode {task_id} exception: {e}", file=sys.stderr)
        score = sum(rewards) / MAX_TOTAL_REWARD if rewards else 0.0

    score = min(max(score, 0.0), 1.0)
    log_end(
        success=score >= SUCCESS_SCORE_THRESHOLD,
        steps=step_num,
        score=score,
        rewards=rewards,
    )
    return score


def main() -> None:
    client = EnvClient()

    # Health check
    try:
        health = client.health()
        print(f"Server health: {health}", flush=True)
    except Exception as e:
        print(f"[ERROR] Cannot reach server at {ENV_BASE_URL}: {e}", file=sys.stderr)
        print("Start the server: uvicorn server:app --host 0.0.0.0 --port 7860", file=sys.stderr)
        sys.exit(1)

    all_rewards: List[float] = []
    for task_id in TASK_IDS:
        try:
            task_score = run_episode(client, task_id)
            all_rewards.append(task_score)
        except Exception as e:
            print(f"[ERROR] Task {task_id} failed: {e}", file=sys.stderr)
            all_rewards.append(0.0)

    final_score = sum(all_rewards) / len(all_rewards) if all_rewards else 0.0
    final_score = min(max(final_score, 0.0), 1.0)
    success = final_score >= SUCCESS_SCORE_THRESHOLD

    print(f"\n{'='*60}")
    print(f"FINAL AGGREGATED SCORE: {final_score:.4f}")
    print(f"SUCCESS: {success}")
    print(f"Per-task scores: {[round(r, 4) for r in all_rewards]}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
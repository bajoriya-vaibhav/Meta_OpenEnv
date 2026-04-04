"""
ChronoVeritas — Baseline inference agent.

Runs all 3 tasks against the environment server (or a local env instance)
and logs results in the [START]/[STEP]/[END] format.

Usage:
    # Start server first:  uvicorn server:app --port 8000
    # Then run:
    export API_BASE_URL=https://api.openai.com/v1
    export MODEL_NAME=gpt-4o-mini
    python inference.py
"""
from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List, Optional

import httpx

# ── Configuration ────────────────────────────────────────────────────

ENV_BASE_URL = os.environ.get("ENV_BASE_URL", "http://localhost:8000")
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o-mini")

TASK_IDS = ["EASY-001", "MED-001", "HARD-001"]
SUCCESS_SCORE_THRESHOLD = 0.5


# ── Logging ──────────────────────────────────────────────────────────

def log(tag: str, data: dict) -> None:
    print(f"[{tag}] {json.dumps(data)}")


# ── LLM Client ──────────────────────────────────────────────────────

def call_llm(prompt: str) -> str:
    """Call the LLM using the openai Python SDK."""
    try:
        from openai import OpenAI

        client = OpenAI(
            base_url=API_BASE_URL,
            api_key=os.environ.get("OPENAI_API_KEY", "sk-placeholder"),
        )
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a fact-checking assistant. Analyze the provided "
                        "documents to determine if a claim is true, false, misleading, "
                        "or unverifiable. Identify which document first introduced a "
                        "mutation and what type of mutation it is.\n"
                        "Respond ONLY in this exact JSON format:\n"
                        '{"verdict": "true|false|misleading|unverifiable", '
                        '"mutation_type": "distortion|omission|fabrication|context_shift|none", '
                        '"mutation_doc_id": "DOC-XXXX", '
                        '"provenance_chain": ["DOC-...", "DOC-..."], '
                        '"confidence": 0.85}'
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=300,
        )
        return response.choices[0].message.content or ""
    except Exception as e:
        print(f"[WARN] LLM call failed: {e}", file=sys.stderr)
        # Return a safe fallback
        return json.dumps({
            "verdict": "unverifiable",
            "mutation_type": "none",
            "mutation_doc_id": None,
            "provenance_chain": [],
            "confidence": 0.1,
        })


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

    def reset(self, task_id: str) -> dict:
        r = self.client.post(f"{self.base_url}/reset", json={"task_id": task_id})
        r.raise_for_status()
        return r.json()

    def step(self, action_type: str, payload: dict) -> dict:
        r = self.client.post(
            f"{self.base_url}/step",
            json={"type": action_type, "payload": payload},
        )
        r.raise_for_status()
        return r.json()


# ── Baseline Agent ───────────────────────────────────────────────────

def run_episode(client: EnvClient, task_id: str) -> float:
    """Run one episode of the baseline agent strategy."""
    log("START", {"task": task_id, "env": "ChronoVeritas", "model": MODEL_NAME})

    rewards: List[float] = []
    step_num = 0

    # 1. Reset
    reset_result = client.reset(task_id)

    # 2. Search with raw claim
    claim = reset_result["observation"]["claim"]
    step_num += 1
    result = client.step("search", {"query": claim})
    reward = result["reward"]
    rewards.append(reward)
    log("STEP", {
        "step": step_num,
        "action": f"search {claim[:50]}...",
        "reward": reward,
        "done": result["done"],
        "error": result["info"].get("error"),
    })

    if result["done"]:
        total = sum(rewards)
        log("END", {"success": total >= SUCCESS_SCORE_THRESHOLD, "steps": step_num, "score": total, "rewards": rewards})
        return total

    # 3. Fetch top 2 docs from search results
    search_results = result.get("info", {}).get("search_results", [])
    corpus_meta = result["observation"].get("corpus_metadata", [])

    # Use corpus metadata if search_results not in info
    docs_to_fetch = []
    if search_results:
        docs_to_fetch = [d["doc_id"] for d in search_results[:2]]
    elif corpus_meta:
        docs_to_fetch = [d["doc_id"] for d in corpus_meta[:2]]

    fetched_contents = []
    for doc_id in docs_to_fetch:
        step_num += 1
        result = client.step("fetch_doc", {"doc_id": doc_id})
        reward = result["reward"]
        rewards.append(reward)
        log("STEP", {
            "step": step_num,
            "action": f"fetch_doc {doc_id}",
            "reward": reward,
            "done": result["done"],
            "error": result["info"].get("error"),
        })

        if result["done"]:
            total = sum(rewards)
            log("END", {"success": total >= SUCCESS_SCORE_THRESHOLD, "steps": step_num, "score": total, "rewards": rewards})
            return total

    # 4. Build prompt and call LLM
    retrieved_docs = result["observation"].get("retrieved_docs", [])
    docs_text = ""
    for doc in retrieved_docs:
        docs_text += f"\n--- {doc['doc_id']}: {doc['title']} (source: {doc['source']}, timestamp: {doc['timestamp']}) ---\n{doc['content']}\n"

    prompt = (
        f"CLAIM: {claim}\n\n"
        f"DOCUMENTS:\n{docs_text}\n\n"
        "Analyze the documents and determine:\n"
        "1. Is the claim true, false, misleading, or unverifiable?\n"
        "2. Which document first introduced the mutation/change?\n"
        "3. What type of mutation is it (distortion, omission, fabrication, context_shift, or none)?\n"
        "4. What is the provenance chain (ordered doc IDs)?\n"
        "Respond in the JSON format specified."
    )

    llm_response = call_llm(prompt)

    # 5. Parse LLM response
    try:
        parsed = json.loads(llm_response)
    except json.JSONDecodeError:
        # Try to extract JSON from response
        import re
        json_match = re.search(r'\{.*\}', llm_response, re.DOTALL)
        if json_match:
            try:
                parsed = json.loads(json_match.group())
            except json.JSONDecodeError:
                parsed = {
                    "verdict": "unverifiable",
                    "mutation_type": "none",
                    "mutation_doc_id": None,
                    "provenance_chain": [],
                    "confidence": 0.1,
                }
        else:
            parsed = {
                "verdict": "unverifiable",
                "mutation_type": "none",
                "mutation_doc_id": None,
                "provenance_chain": [],
                "confidence": 0.1,
            }

    # 6. set_mutation_point
    mutation_doc_id = parsed.get("mutation_doc_id")
    mutation_type = parsed.get("mutation_type", "none")

    if mutation_doc_id:
        result = client.step("set_mutation_point", {
            "doc_id": mutation_doc_id,
            "mutation_type": mutation_type,
        })
        reward = result["reward"]
        rewards.append(reward)
        step_num += 1
        log("STEP", {
            "step": step_num,
            "action": f"set_mutation_point {mutation_doc_id} {mutation_type}",
            "reward": reward,
            "done": result["done"],
            "error": result["info"].get("error"),
        })

        if result["done"]:
            total = sum(rewards)
            log("END", {"success": total >= SUCCESS_SCORE_THRESHOLD, "steps": step_num, "score": total, "rewards": rewards})
            return total

    # 7. submit_verdict
    result = client.step("submit_verdict", {
        "verdict": parsed.get("verdict", "unverifiable"),
        "mutation_type": mutation_type,
        "mutation_doc_id": mutation_doc_id,
        "provenance_chain": parsed.get("provenance_chain", []),
        "confidence": parsed.get("confidence", 0.5),
    })
    reward = result["reward"]
    rewards.append(reward)
    step_num += 1
    log("STEP", {
        "step": step_num,
        "action": f"submit_verdict {parsed.get('verdict', 'unverifiable')}",
        "reward": reward,
        "done": result["done"],
        "error": result["info"].get("error"),
    })

    total = sum(rewards)
    log("END", {
        "success": total >= SUCCESS_SCORE_THRESHOLD,
        "steps": step_num,
        "score": round(total, 4),
        "rewards": [round(r, 4) for r in rewards],
    })

    return total


def main() -> None:
    client = EnvClient()

    # Health check
    try:
        health = client.health()
        print(f"Server health: {health}")
    except Exception as e:
        print(f"[ERROR] Cannot reach server at {ENV_BASE_URL}: {e}", file=sys.stderr)
        print("Start the server first: uvicorn server:app --port 8000", file=sys.stderr)
        sys.exit(1)

    all_rewards: List[float] = []

    for task_id in TASK_IDS:
        try:
            score = run_episode(client, task_id)
            all_rewards.append(score)
        except Exception as e:
            print(f"[ERROR] Task {task_id} failed: {e}", file=sys.stderr)
            all_rewards.append(0.0)

    # Final aggregated score
    MAX_TOTAL_REWARD = 3.0
    final_score = sum(all_rewards) / MAX_TOTAL_REWARD
    final_score = min(max(final_score, 0.0), 1.0)
    success = final_score >= SUCCESS_SCORE_THRESHOLD

    print(f"\n{'='*60}")
    print(f"FINAL AGGREGATED SCORE: {final_score:.4f}")
    print(f"SUCCESS: {success}")
    print(f"Per-task: {[round(r, 4) for r in all_rewards]}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

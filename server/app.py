"""
ChronoVeritas — FastAPI Server (v2).

Includes all v1 OpenEnv endpoints (reset, step, state, health)
plus v2 endpoints for the multi-agent story (mutate, spread, demo).
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import uvicorn
import yaml
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import RedirectResponse, JSONResponse
from pydantic import BaseModel

# Ensure project root is on path for agents imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from env.environment import ChronoVeritasEnv
from env.models import Action, Observation, StepResult

import os

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, "openenv.yaml")

with open(CONFIG_PATH, "r") as f:
    OPENENV_CONFIG = yaml.safe_load(f)

app = FastAPI(
    title="ChronoVeritas",
    description="Claim lifecycle verification environment (OpenEnv) — v2 with Mutator + Spreader endpoints",
    version="2.0.0",
)

# Global environment instance
env = ChronoVeritasEnv()


# ── Request Models ───────────────────────────────────────────────────

class ResetRequest(BaseModel):
    task_id: Optional[str] = None


class MutateRequest(BaseModel):
    """Request body for POST /mutate."""
    fact_id: Optional[str] = None
    mutation_type: Optional[str] = None  # distortion | fabrication | omission | context_shift
    seed: Optional[int] = None


class SpreadRequest(BaseModel):
    """Request body for POST /spread — takes mutation result and builds a task."""
    mutation_type: str = "distortion"
    false_claim: str = ""
    true_claim: str = ""
    original_content: str = ""
    mutated_content: str = ""
    difficulty: str = "easy"
    fact_id: Optional[str] = None
    seed: Optional[int] = None


# ══════════════════════════════════════════════════════════════════════
#  V1 ENDPOINTS (kept intact — do NOT modify)
# ══════════════════════════════════════════════════════════════════════

@app.get("/")
def root():
    return RedirectResponse(url="/docs")

@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "healthy"}


@app.get("/tasks")
async def list_tasks() -> List[Dict[str, Any]]:
    return env.get_task_list()


@app.post("/reset", response_model=StepResult)
async def reset(req: ResetRequest = ResetRequest()) -> StepResult:
    try:
        result = await env.reset(task_id=req.task_id)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/step", response_model=StepResult)
async def step(action: Action) -> StepResult:
    if env.state is None:
        raise HTTPException(status_code=400, detail="Call /reset first")
    result = await env.step(action)
    return result


@app.get("/state", response_model=Observation)
async def get_state() -> Observation:
    if env.state is None:
        raise HTTPException(status_code=400, detail="Call /reset first")
    return await env.get_state()

@app.get("/metadata")
def metadata():
    return {
        "name": OPENENV_CONFIG["name"],
        "description": OPENENV_CONFIG["description"],
        "version": OPENENV_CONFIG.get("version", "unknown")
    }

@app.get("/schema")
def schema():
    return {
        "action": {
            "type": "string",
            "enum": OPENENV_CONFIG["actions"]
        },
        "observation": {
            "type": "object",
            "description": "Environment response after each step"
        },
        "state": {
            "type": "object",
            "description": "Internal environment state"
        }
    }

@app.post("/mcp")
async def mcp(request: Request):
    body = await request.json()

    return {
        "jsonrpc": "2.0",
        "id": body.get("id"),
        "result": {
            "available_actions": OPENENV_CONFIG["actions"],
            "message": "MCP connected"
        }
    }


# ══════════════════════════════════════════════════════════════════════
#  V2 ENDPOINTS — Mutator, Spreader, Demo
# ══════════════════════════════════════════════════════════════════════

@app.post("/mutate")
async def mutate_endpoint(req: MutateRequest) -> Dict[str, Any]:
    """
    POST /mutate — Takes a fact_id + mutation_type, returns mutated content.

    If fact_id is None, a random seed fact is selected.
    If mutation_type is None, the Mutator picks randomly.
    """
    try:
        from agents.task_bank import SEED_FACT_MAP, SEED_FACTS, get_random_fact
        from agents.mutator import Mutator
        import random

        rng = random.Random(req.seed)
        mutator = Mutator(seed=req.seed)

        # Select fact
        if req.fact_id and req.fact_id in SEED_FACT_MAP:
            fact = SEED_FACT_MAP[req.fact_id]
        else:
            fact = get_random_fact(rng=rng)

        # Apply mutation
        result = mutator.mutate(fact, mutation_type=req.mutation_type)

        return {
            "status": "success",
            "fact_id": fact.fact_id,
            "mutation_type": result.mutation_type,
            "true_claim": result.true_claim,
            "false_claim": result.false_claim,
            "original_content": result.original_content[:500],
            "mutated_content": result.mutated_content[:500],
            "diff_description": result.diff_description,
            "original_number": result.original_number,
            "mutated_number": result.mutated_number,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Mutation failed: {str(e)}")


@app.post("/spread")
async def spread_endpoint(req: SpreadRequest) -> Dict[str, Any]:
    """
    POST /spread — Takes a mutation description and builds a full TaskSpec.

    If fact_id is provided, mutates that fact first; otherwise uses the
    provided mutation fields directly.
    """
    try:
        from agents.task_bank import SEED_FACT_MAP, get_random_fact
        from agents.mutator import Mutator
        from agents.spreader import Spreader
        import random

        rng = random.Random(req.seed)
        mutator = Mutator(seed=req.seed)
        spreader = Spreader(seed=req.seed)

        # Get fact
        if req.fact_id and req.fact_id in SEED_FACT_MAP:
            fact = SEED_FACT_MAP[req.fact_id]
        else:
            fact = get_random_fact(rng=rng)

        # Generate mutation
        mutation = mutator.mutate(fact, mutation_type=req.mutation_type)

        # Build full task
        task = spreader.spread(mutation, difficulty=req.difficulty)

        return {
            "status": "success",
            "task": task,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Spread failed: {str(e)}")


@app.get("/demo")
async def demo_endpoint() -> Dict[str, Any]:
    """
    GET /demo — Returns a real-time demo of all 3 agents in sequence:
    1. Mutator distorts a fact
    2. Spreader builds a corpus
    3. Shows the full TaskSpec ready for the Fact-Checker

    This powers the HF Spaces demo UI.
    """
    try:
        from agents.task_bank import SEED_FACTS, get_random_fact
        from agents.mutator import Mutator
        from agents.spreader import Spreader
        import random
        import time

        seed = int(time.time()) % 10000
        rng = random.Random(seed)
        mutator = Mutator(seed=seed)
        spreader = Spreader(seed=seed)

        # Step 1: Pick a random fact
        fact = get_random_fact(rng=rng)

        # Step 2: Mutate it
        mutation_type = rng.choice(["distortion", "fabrication", "omission", "context_shift"])
        mutation = mutator.mutate(fact, mutation_type=mutation_type)

        # Step 3: Spread it into a task
        difficulty = rng.choice(["easy", "medium"])
        task = spreader.spread(mutation, difficulty=difficulty)

        return {
            "status": "success",
            "seed": seed,
            "story": {
                "step_1_mutator": {
                    "description": "Mutator takes a true claim and distorts it",
                    "fact_id": fact.fact_id,
                    "domain": fact.domain,
                    "true_claim": mutation.true_claim,
                    "mutation_type": mutation.mutation_type,
                    "false_claim": mutation.false_claim,
                    "diff": mutation.diff_description,
                },
                "step_2_spreader": {
                    "description": "Spreader builds a corpus of documents embedding the distorted claim",
                    "difficulty": difficulty,
                    "corpus_size": len(task["corpus"]),
                    "mutation_doc_id": task["ground_truth"]["gt_mutation_doc_id"],
                    "provenance_chain": task["ground_truth"]["gt_provenance_chain"],
                },
                "step_3_fact_checker": {
                    "description": "Fact-Checker (GRPO-trained LLM) investigates and identifies the mutation",
                    "claim_to_investigate": task["claim"],
                    "documents_available": len(task["corpus"]),
                    "expected_verdict": task["ground_truth"]["gt_verdict"],
                    "expected_mutation_type": task["ground_truth"]["gt_mutation_type"],
                },
            },
            "full_task": task,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Demo failed: {str(e)}")


@app.get("/seed-facts")
async def list_seed_facts() -> List[Dict[str, Any]]:
    """List all available seed facts for the /mutate endpoint."""
    try:
        from agents.task_bank import SEED_FACTS
        return [
            {
                "fact_id": f.fact_id,
                "domain": f.domain,
                "true_claim": f.true_claim,
                "true_entity": f.true_entity,
                "true_number": f.true_number,
            }
            for f in SEED_FACTS
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def main():
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
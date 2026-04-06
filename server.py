"""
ChronoVeritas — FastAPI Server.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from env.environment import ChronoVeritasEnv
from env.models import Action, Observation, StepResult

app = FastAPI(
    title="ChronoVeritas",
    description="Claim lifecycle verification environment (OpenEnv)",
    version="1.0.0",
)

# Global environment instance
env = ChronoVeritasEnv()


# ── Request Models ───────────────────────────────────────────────────

class ResetRequest(BaseModel):
    task_id: Optional[str] = None


# ── Endpoints ────────────────────────────────────────────────────────

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

"""
ChronoVeritas — FastAPI Server.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import RedirectResponse

from pydantic import BaseModel

from env.environment import ChronoVeritasEnv
from env.models import Action, Observation, StepResult

import yaml

import os

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, "openenv.yaml")

with open(CONFIG_PATH, "r") as f:
    OPENENV_CONFIG = yaml.safe_load(f)

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


def main():
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
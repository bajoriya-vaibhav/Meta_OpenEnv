"""
ChronoVeritas — All Pydantic v2 typed models.
"""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, Field


# ── Document & Metadata ──────────────────────────────────────────────

class DocMeta(BaseModel):
    """Lightweight document stub returned by search."""
    doc_id: str
    title: str
    source: str
    timestamp: int
    tags: List[str] = Field(default_factory=list)
    snippet: str = ""


class Document(BaseModel):
    """Full document including content."""
    doc_id: str
    title: str
    source: str
    timestamp: int
    tags: List[str] = Field(default_factory=list)
    snippet: str = ""
    content: str = ""


# ── Timeline & Contradiction ─────────────────────────────────────────

class TimelineEntry(BaseModel):
    doc_id: str
    event_label: str
    timestamp: Optional[int] = None


class MutationDecl(BaseModel):
    """Agent's declared mutation point (before final verdict)."""
    doc_id: str
    mutation_type: Literal[
        "distortion", "omission", "fabrication", "context_shift", "none"
    ]


# ── Observation ──────────────────────────────────────────────────────

class Observation(BaseModel):
    claim: str = ""
    corpus_metadata: List[DocMeta] = Field(default_factory=list)
    retrieved_docs: List[Document] = Field(default_factory=list)
    agent_timeline: List[TimelineEntry] = Field(default_factory=list)
    flagged_contradictions: List[Tuple[str, str]] = Field(default_factory=list)
    current_step: int = 0
    max_steps: int = 15
    token_budget_remaining: int = 8000
    partial_reward_so_far: float = 0.0


# ── Action ───────────────────────────────────────────────────────────

class Action(BaseModel):
    type: Literal[
        "search",
        "fetch_doc",
        "add_timeline_event",
        "flag_contradiction",
        "set_mutation_point",
        "submit_verdict",
    ]
    payload: Dict[str, Any] = Field(default_factory=dict)


class VerdictPayload(BaseModel):
    verdict: Literal["true", "false", "misleading", "unverifiable"]
    mutation_type: Literal[
        "distortion", "omission", "fabrication", "context_shift", "none"
    ]
    mutation_doc_id: Optional[str] = None
    provenance_chain: List[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)


# ── Step / Grade Results ─────────────────────────────────────────────

class StepResult(BaseModel):
    observation: Observation
    reward: float = 0.0
    done: bool = False
    info: Dict[str, Any] = Field(default_factory=dict)


class GradeResult(BaseModel):
    total: float
    breakdown: Dict[str, float] = Field(default_factory=dict)


# ── Task Specification ───────────────────────────────────────────────

class GroundTruth(BaseModel):
    gt_verdict: Literal["true", "false", "misleading", "unverifiable"]
    gt_mutation_type: Literal[
        "distortion", "omission", "fabrication", "context_shift", "none"
    ]
    gt_mutation_doc_id: Optional[str] = None
    gt_provenance_chain: List[str] = Field(default_factory=list)
    gt_timeline: List[str] = Field(default_factory=list)


class TaskSpec(BaseModel):
    task_id: str
    difficulty: Literal["easy", "medium", "hard"]
    max_steps: int
    claim: str
    ground_truth: GroundTruth
    corpus: List[Document] = Field(default_factory=list)


# ── Episode State (not a Pydantic model — mutable runtime object) ───
# Defined in state_manager.py

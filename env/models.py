"""
ChronoVeritas — All Pydantic v2 typed models.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, Field, field_validator, model_validator


# ── Shared literals ──────────────────────────────────────────────────────────

MutationType = Literal["distortion", "omission", "fabrication", "context_shift", "none"]
VerdictType = Literal["true", "false", "misleading", "unverifiable"]
ActionType = Literal[
    "search",
    "fetch_doc",
    "add_timeline_event",
    "flag_contradiction",
    "set_mutation_point",
    "submit_verdict",
]


# ── Document & Metadata ──────────────────────────────────────────────────────

class DocMeta(BaseModel):
    """Lightweight document stub returned by search."""

    doc_id: str
    title: str
    source: str
    timestamp: int  # Unix epoch seconds
    reliability_tier: int = Field(default=2, ge=1, le=3)  # 1=official, 2=institutional, 3=informal
    tags: List[str] = Field(default_factory=list)
    snippet: str = ""

    @field_validator("doc_id", "title", "source")
    @classmethod
    def non_empty_string(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Field must be a non-empty string")
        return v.strip()

    @field_validator("timestamp")
    @classmethod
    def positive_timestamp(cls, v: int) -> int:
        if v < 0:
            raise ValueError(f"timestamp must be non-negative, got {v}")
        return v

    @field_validator("tags")
    @classmethod
    def dedupe_tags(cls, v: List[str]) -> List[str]:
        # Preserve insertion order while deduplicating
        seen: set[str] = set()
        return [t for t in v if not (t in seen or seen.add(t))]  # type: ignore[func-returns-value]

    @property
    def age_seconds(self) -> int:
        """How many seconds ago this document was timestamped."""
        return max(0, int(time.time()) - self.timestamp)

    def slim_dump(self) -> Dict[str, Any]:
        """Return only the fields an agent typically needs at search time."""
        return {
            "doc_id": self.doc_id,
            "title": self.title,
            "source": self.source,
            "timestamp": self.timestamp,
            "reliability_tier": self.reliability_tier,
            "tags": self.tags,
            "snippet": self.snippet,
        }


class Document(DocMeta):
    """Full document including content. Extends DocMeta so it IS-A DocMeta."""

    content: str = ""

    def to_meta(self) -> DocMeta:
        """Strip content and return the lightweight metadata view."""
        return DocMeta(
            doc_id=self.doc_id,
            title=self.title,
            source=self.source,
            timestamp=self.timestamp,
            reliability_tier=self.reliability_tier,
            tags=list(self.tags),
            snippet=self.snippet,
        )

    @property
    def estimated_tokens(self) -> int:
        """Rough token estimate (4 chars ≈ 1 token)."""
        return len(self.content) // 4


# ── Timeline & Contradiction ─────────────────────────────────────────────────

class TimelineEntry(BaseModel):
    doc_id: str
    event_label: str
    timestamp: Optional[int] = None

    @field_validator("doc_id", "event_label")
    @classmethod
    def non_empty_string(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Field must be a non-empty string")
        return v.strip()


class MutationDecl(BaseModel):
    """Agent's declared mutation point (before final verdict)."""

    doc_id: str
    mutation_type: MutationType

    @field_validator("doc_id")
    @classmethod
    def non_empty_string(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("doc_id must be non-empty")
        return v.strip()


# ── Observation ──────────────────────────────────────────────────────────────

class Observation(BaseModel):
    claim: str = ""
    corpus_metadata: List[DocMeta] = Field(default_factory=list)
    retrieved_docs: List[Document] = Field(default_factory=list)
    agent_timeline: List[TimelineEntry] = Field(default_factory=list)
    flagged_contradictions: List[Tuple[str, str]] = Field(default_factory=list)
    current_step: int = 0
    max_steps: int = 15
    token_budget_remaining: int = 8_000
    partial_reward_so_far: float = 0.0

    @property
    def steps_remaining(self) -> int:
        return max(self.max_steps - self.current_step, 0)

    @property
    def is_budget_critical(self) -> bool:
        """True when less than 20 % of token budget or steps remain."""
        token_pct = self.token_budget_remaining / max(8_000, 1)
        step_pct = self.steps_remaining / max(self.max_steps, 1)
        return token_pct < 0.20 or step_pct < 0.20


# ── Action ────────────────────────────────────────────────────────────────────

class Action(BaseModel):
    type: ActionType
    payload: Dict[str, Any] = Field(default_factory=dict)

    # ── Convenience factory methods ──────────────────────────────────

    @classmethod
    def search(cls, query: str, *, date_from: Optional[int] = None, date_to: Optional[int] = None) -> "Action":
        payload: Dict[str, Any] = {"query": query}
        if date_from is not None:
            payload["date_from"] = date_from
        if date_to is not None:
            payload["date_to"] = date_to
        return cls(type="search", payload=payload)

    @classmethod
    def fetch_doc(cls, doc_id: str) -> "Action":
        return cls(type="fetch_doc", payload={"doc_id": doc_id})

    @classmethod
    def add_timeline_event(
        cls, doc_id: str, event_label: str, timestamp: Optional[int] = None
    ) -> "Action":
        payload: Dict[str, Any] = {"doc_id": doc_id, "event_label": event_label}
        if timestamp is not None:
            payload["timestamp"] = timestamp
        return cls(type="add_timeline_event", payload=payload)

    @classmethod
    def flag_contradiction(cls, doc_id_a: str, doc_id_b: str) -> "Action":
        return cls(
            type="flag_contradiction",
            payload={"doc_id_a": doc_id_a, "doc_id_b": doc_id_b},
        )

    @classmethod
    def set_mutation_point(cls, doc_id: str, mutation_type: MutationType) -> "Action":
        return cls(
            type="set_mutation_point",
            payload={"doc_id": doc_id, "mutation_type": mutation_type},
        )

    @classmethod
    def submit_verdict(
        cls,
        verdict: VerdictType,
        mutation_type: MutationType,
        provenance_chain: List[str],
        *,
        mutation_doc_id: Optional[str] = None,
        confidence: float = 0.5,
    ) -> "Action":
        return cls(
            type="submit_verdict",
            payload={
                "verdict": verdict,
                "mutation_type": mutation_type,
                "mutation_doc_id": mutation_doc_id,
                "provenance_chain": provenance_chain,
                "confidence": confidence,
            },
        )


class VerdictPayload(BaseModel):
    verdict: VerdictType
    mutation_type: MutationType
    mutation_doc_id: Optional[str] = None
    provenance_chain: List[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)

    @model_validator(mode="after")
    def mutation_doc_required_when_not_none(self) -> "VerdictPayload":
        if self.mutation_type != "none" and not self.mutation_doc_id:
            raise ValueError(
                "mutation_doc_id must be provided when mutation_type is not 'none'"
            )
        return self

    @field_validator("confidence")
    @classmethod
    def clamp_confidence(cls, v: float) -> float:
        return max(0.0, min(1.0, v))

    @property
    def is_mutation_claimed(self) -> bool:
        return self.mutation_type != "none"


# ── Step / Grade Results ─────────────────────────────────────────────────────

class StepResult(BaseModel):
    observation: Observation
    reward: float = 0.0
    done: bool = False
    info: Dict[str, Any] = Field(default_factory=dict)

    @property
    def has_error(self) -> bool:
        return "error" in self.info

    @property
    def error_message(self) -> Optional[str]:
        return self.info.get("error")


class GradeResult(BaseModel):
    total: float
    breakdown: Dict[str, float] = Field(default_factory=dict)

    @property
    def capped_total(self) -> float:
        """Clamp to [0, 1] regardless of intermediate floating-point drift."""
        return max(0.0, min(1.0, self.total))


# ── Task Specification ────────────────────────────────────────────────────────

class GroundTruth(BaseModel):
    gt_verdict: VerdictType
    gt_mutation_type: MutationType
    gt_mutation_doc_id: Optional[str] = None
    gt_provenance_chain: List[str] = Field(default_factory=list)
    gt_timeline: List[str] = Field(default_factory=list)
    gt_conflict_fields: List[str] = Field(default_factory=list)  # hard tasks only

    @model_validator(mode="after")
    def mutation_doc_required_when_not_none(self) -> "GroundTruth":
        if self.gt_mutation_type != "none" and not self.gt_mutation_doc_id:
            raise ValueError(
                "gt_mutation_doc_id must be set when gt_mutation_type is not 'none'"
            )
        return self


class TaskSpec(BaseModel):
    task_id: str
    difficulty: Literal["easy", "medium", "hard"]
    max_steps: int = Field(ge=1)
    claim: str
    ground_truth: GroundTruth
    corpus: List[Document] = Field(default_factory=list)

    @field_validator("task_id", "claim")
    @classmethod
    def non_empty_string(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Field must be a non-empty string")
        return v.strip()

    def corpus_by_id(self) -> Dict[str, Document]:
        return {doc.doc_id: doc for doc in self.corpus}
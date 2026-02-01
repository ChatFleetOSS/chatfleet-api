# generated-by: codex-agent 2025-02-15T00:16:00Z
"""
Background job schemas.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

from .common import UUIDStr

JobType = Literal["RAG_INDEX", "RAG_REBUILD", "RAG_RESET", "CHAT_COMPLETION"]
JobState = Literal["queued", "running", "done", "error"]
JobPhase = Literal["queued", "chunking", "embedding", "indexing", "suggestions", "finalizing"]


class JobProgressTotals(BaseModel):
    docs_total: int = Field(ge=0)
    docs_done: int = Field(ge=0)
    chunks_total: int = Field(ge=0)
    chunks_done: int = Field(ge=0)


class JobAccepted(BaseModel):
    job_id: UUIDStr
    corr_id: str


class JobStatusResponse(BaseModel):
    job_id: UUIDStr
    type: JobType
    status: JobState
    progress: float = Field(ge=0, le=1)
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    result: Optional[dict[str, Any]] = None
    phase: Optional[JobPhase] = None
    totals: Optional[JobProgressTotals] = None
    suggestions_ready: bool = False
    error: Optional[str] = None
    corr_id: str

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
    error: Optional[str] = None
    corr_id: str

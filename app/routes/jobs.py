# generated-by: codex-agent 2025-02-15T00:27:00Z
"""
Background job polling endpoint.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Path, status

from app.dependencies.auth import get_current_user
from app.models.envelope import ErrorEnvelope
from app.models.jobs import JobStatusResponse
from app.services.jobs import job_manager
from app.utils.responses import raise_http_error

router = APIRouter(prefix="/jobs", tags=["Jobs"])


@router.get("/{job_id}", response_model=JobStatusResponse)
async def job_status(job_id: str = Path(...), current_user = Depends(get_current_user)) -> JobStatusResponse:
    job = job_manager.get(job_id)
    if not job:
        raise_http_error("JOB_NOT_FOUND", f"Job '{job_id}' not found", status.HTTP_404_NOT_FOUND)
    return job

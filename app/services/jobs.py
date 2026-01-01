# generated-by: codex-agent 2025-02-15T00:18:00Z
"""
Minimal in-memory job manager used to orchestrate asynchronous tasks.

This keeps us compliant with the spec requirement that long-running work
executed by background jobs returns a pollable `job_id`.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Awaitable, Callable, Dict, Optional
from uuid import uuid4

from app.core.corr_id import get_corr_id
from app.models.jobs import JobState, JobStatusResponse, JobType, JobProgressTotals, JobPhase

logger = logging.getLogger(__name__)


@dataclass
class JobRecord:
    job_id: str
    type: JobType
    status: JobState = "queued"
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    progress: float = 0.0
    result: Optional[dict] = None
    phase: Optional[JobPhase] = None
    totals: Optional[JobProgressTotals] = None
    error: Optional[str] = None
    corr_id: str = field(default_factory=get_corr_id)


Runner = Callable[[JobRecord], Awaitable[None]]


class JobManager:
    """In-memory async job manager."""

    def __init__(self) -> None:
        self._jobs: Dict[str, JobRecord] = {}

    def _store(self, record: JobRecord) -> None:
        self._jobs[record.job_id] = record

    def get(self, job_id: str) -> Optional[JobStatusResponse]:
        record = self._jobs.get(job_id)
        if not record:
            return None
        return JobStatusResponse(
            job_id=record.job_id,
            type=record.type,
            status=record.status,
            progress=record.progress,
            started_at=record.started_at,
            finished_at=record.finished_at,
            result=record.result,
            phase=record.phase,
            totals=record.totals,
            error=record.error,
            corr_id=record.corr_id,
        )

    def schedule(self, job_type: JobType, runner: Runner) -> JobRecord:
        job_id = str(uuid4())
        record = JobRecord(job_id=job_id, type=job_type)
        self._store(record)

        async def execute() -> None:
            logger.info("Job %s (%s) started", record.job_id, record.type)
            record.status = "running"
            record.started_at = datetime.now(timezone.utc)
            try:
                await runner(record)
                if record.status != "error":
                    record.status = "done"
                record.progress = 1.0
            except Exception as exc:  # pragma: no cover — defensive
                logger.exception("Job %s failed", record.job_id)
                record.status = "error"
                record.error = str(exc)
            finally:
                record.finished_at = datetime.now(timezone.utc)
                logger.info("Job %s finished with status %s", record.job_id, record.status)

        asyncio.create_task(execute())
        return record


job_manager = JobManager()

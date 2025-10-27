# generated-by: codex-agent 2025-10-27T00:00:00Z
"""
Healthcheck endpoint for container orchestration.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter

from app.core.config import settings
from app.core.database import get_database

router = APIRouter(prefix="/health", tags=["Health"])


@router.get("")
async def health() -> dict[str, str | bool]:
    # Check Mongo connectivity
    try:
        db = get_database()
        await db.command("ping")
        mongo_ok = True
    except Exception:
        mongo_ok = False

    # Check storage directories exist (do not create/write here)
    index_ok = Path(settings.index_dir).exists()
    upload_ok = Path(settings.upload_dir).exists()

    status = mongo_ok and index_ok and upload_ok
    return {
        "status": "ok" if status else "degraded",
        "mongo": mongo_ok,
        "index_dir": index_ok,
        "upload_dir": upload_ok,
    }


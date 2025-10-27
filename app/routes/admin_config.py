# generated-by: codex-agent 2025-02-15T00:28:00Z
"""
Admin configuration endpoint.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from app.core.config import settings
from app.dependencies.auth import require_admin
from app.models.admin import AdminConfigResponse
from app.utils.responses import with_corr_id

router = APIRouter(prefix="/admin/config", tags=["Admin Config"])


@router.get("", response_model=AdminConfigResponse)
async def admin_config(current_user = Depends(require_admin)) -> AdminConfigResponse:
    payload = {
        "chat_model": settings.chat_model,
        "embed_model": settings.embed_model,
        "index_dir": str(settings.index_dir),
        "upload_dir": str(settings.upload_dir),
        "max_upload_mb": settings.max_upload_mb,
    }
    return AdminConfigResponse(**with_corr_id(payload))

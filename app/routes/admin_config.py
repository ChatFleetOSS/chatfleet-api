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
from app.services.runtime_config import get_llm_config

router = APIRouter(prefix="/admin/config", tags=["Admin Config"])


@router.get("", response_model=AdminConfigResponse)
async def admin_config(current_user = Depends(require_admin)) -> AdminConfigResponse:
    # Prefer runtime overrides for models so the dashboard reflects what will be used at runtime.
    cfg = await get_llm_config()
    from app.services.runtime_config import get_runtime_overrides_sync
    index_dir, upload_dir, max_upload_mb, _, _ = get_runtime_overrides_sync()
    payload = {
        "chat_model": cfg.chat_model,
        "embed_model": cfg.embed_model,
        "index_dir": str(index_dir),
        "upload_dir": str(upload_dir),
        "max_upload_mb": int(max_upload_mb),
    }
    return AdminConfigResponse(**with_corr_id(payload))

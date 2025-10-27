# generated-by: codex-agent 2025-02-15T00:25:00Z
"""
RAG listing APIs (user + admin variants).
"""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, Query

from app.dependencies.auth import get_current_user, require_admin
from app.models.rag import RagListResponse
from app.services.rags import list_all_rags, list_rags_for_slugs
from app.utils.responses import with_corr_id

router = APIRouter(tags=["RAG"])


@router.get("/rag/list", response_model=RagListResponse)
async def rag_list(
    limit: int = Query(50, ge=1, le=200),
    cursor: Optional[str] = Query(default=None),
    current_user = Depends(get_current_user),
) -> RagListResponse:
    if current_user.get("role") == "admin":
        items, next_cursor = await list_all_rags(limit=limit, cursor=cursor)
    else:
        items, next_cursor = await list_rags_for_slugs(current_user.get("rags", []), limit=limit, cursor=cursor)
    payload = {"items": items, "next_cursor": next_cursor}
    return RagListResponse(**with_corr_id(payload))


@router.get("/admin/rag/list", response_model=RagListResponse)
async def rag_list_admin(
    limit: int = Query(50, ge=1, le=200),
    cursor: Optional[str] = Query(default=None),
    admin_user = Depends(require_admin),
) -> RagListResponse:
    items, next_cursor = await list_all_rags(limit=limit, cursor=cursor)
    payload = {"items": items, "next_cursor": next_cursor}
    return RagListResponse(**with_corr_id(payload))

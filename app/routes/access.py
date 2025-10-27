# generated-by: codex-agent 2025-02-15T00:27:00Z
"""
Access control routes for managing RAG memberships.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, status

from app.dependencies.auth import require_admin
from app.models.rag import RagUserUpsertRequest, RagUserUpsertResponse, RagUsersResponse
from app.services.rags import add_user_to_rag, list_rag_users, remove_user_from_rag
from app.services.users import find_user_by_email, find_user_by_id
from app.utils.responses import raise_http_error, with_corr_id

router = APIRouter(tags=["Access Control"])


@router.get("/rag/users", response_model=RagUsersResponse)
async def rag_users(
    rag_slug: str,
    admin_user = Depends(require_admin),
) -> RagUsersResponse:
    return await list_rag_users(rag_slug)


@router.post("/rag/users/add", response_model=RagUserUpsertResponse)
async def rag_users_add(
    payload: RagUserUpsertRequest,
    admin_user = Depends(require_admin),
) -> RagUserUpsertResponse:
    user = None
    if payload.user_id:
        user = await find_user_by_id(payload.user_id)
    elif payload.email:
        user = await find_user_by_email(payload.email)
    if not user:
        raise_http_error("USER_NOT_FOUND", "Target user not found", status.HTTP_404_NOT_FOUND)

    await add_user_to_rag(payload.rag_slug, user["_id"], user.get("rags", []))
    return RagUserUpsertResponse(**with_corr_id({"rag_slug": payload.rag_slug, "user_id": str(user["_id"])}))


@router.post("/rag/users/remove", response_model=RagUserUpsertResponse)
async def rag_users_remove(
    payload: RagUserUpsertRequest,
    admin_user = Depends(require_admin),
) -> RagUserUpsertResponse:
    user = None
    if payload.user_id:
        user = await find_user_by_id(payload.user_id)
    elif payload.email:
        user = await find_user_by_email(payload.email)
    if not user:
        raise_http_error("USER_NOT_FOUND", "Target user not found", status.HTTP_404_NOT_FOUND)

    await remove_user_from_rag(payload.rag_slug, user["_id"], user.get("rags", []))
    return RagUserUpsertResponse(**with_corr_id({"rag_slug": payload.rag_slug, "user_id": str(user["_id"])}))

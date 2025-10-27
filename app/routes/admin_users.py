# generated-by: codex-agent 2025-02-15T00:24:00Z
"""
Admin user management routes.
"""

from __future__ import annotations

from secrets import token_urlsafe
from typing import Optional

from fastapi import APIRouter, Depends, Query, status
from pymongo.errors import DuplicateKeyError

from app.dependencies.auth import get_current_user, require_admin
from app.models.auth import AdminCreateUserRequest, UserPublic, UsersListResponse
from app.services.logging import write_system_log
from app.services.users import create_user_from_admin, list_users, user_to_public
from app.utils.responses import raise_http_error, with_corr_id

router = APIRouter(prefix="/admin/users", tags=["Users (Admin)"])


@router.get("", response_model=UsersListResponse)
async def admin_list_users(
    limit: int = Query(50, ge=1, le=200),
    cursor: Optional[str] = Query(default=None),
    current_user = Depends(require_admin),
) -> UsersListResponse:
    items, next_cursor = await list_users(limit=limit, cursor=cursor)
    await write_system_log(event="admin.users.list", user_id=str(current_user["_id"]), details={"limit": limit})
    return UsersListResponse(**with_corr_id({"items": items, "next_cursor": next_cursor}))


@router.post("", response_model=UserPublic, status_code=status.HTTP_201_CREATED)
async def admin_create_user(
    payload: AdminCreateUserRequest,
    current_user = Depends(require_admin),
) -> UserPublic:
    generated_password = token_urlsafe(12)
    try:
        doc = await create_user_from_admin(payload, generated_password)
    except DuplicateKeyError:
        raise_http_error("EMAIL_EXISTS", "Email already registered", status.HTTP_400_BAD_REQUEST)
    await write_system_log(
        event="admin.users.create",
        user_id=str(current_user["_id"]),
        details={"target_user": payload.email, "role": payload.role},
    )
    return user_to_public(doc)

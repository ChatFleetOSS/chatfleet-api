# generated-by: codex-agent 2025-02-15T00:23:00Z
"""
Auth routes: register, login, and me.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, status
from pymongo.errors import DuplicateKeyError

from app.dependencies.auth import authenticate_user, create_access_token, get_current_user
from app.models.auth import AuthResponse, LoginRequest, RegisterRequest, UserPublic
from app.services.users import create_user_from_register, user_to_public
from app.utils.responses import raise_http_error, with_corr_id
from app.utils.responses import with_corr_id

router = APIRouter(prefix="/auth", tags=["Auth"])


@router.post("/register", response_model=AuthResponse, status_code=status.HTTP_201_CREATED)
async def register(payload: RegisterRequest) -> AuthResponse:
    try:
        doc = await create_user_from_register(payload)
    except DuplicateKeyError:
        raise_http_error("EMAIL_EXISTS", "Email already registered", status.HTTP_400_BAD_REQUEST)
    token = create_access_token(doc)
    payload_dict = {"token": token, "user": user_to_public(doc)}
    return AuthResponse(**with_corr_id(payload_dict))


@router.post("/login", response_model=AuthResponse)
async def login(payload: LoginRequest) -> AuthResponse:
    user = await authenticate_user(payload.email, payload.password)
    token = create_access_token(user)
    payload_dict = {"token": token, "user": user_to_public(user)}
    return AuthResponse(**with_corr_id(payload_dict))


@router.get("/me", response_model=UserPublic)
async def me(current_user = Depends(get_current_user)) -> UserPublic:
    return user_to_public(current_user)

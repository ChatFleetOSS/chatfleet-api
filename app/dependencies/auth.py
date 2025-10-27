# generated-by: codex-agent 2025-02-15T00:22:00Z
"""
JWT helpers and FastAPI dependencies for auth + RBAC.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict

import jwt
from bson import ObjectId
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel

from app.core.config import settings
from app.core.corr_id import get_corr_id
from app.models.auth import UserPublic
from app.services.users import (
    find_user_by_email,
    find_user_by_id,
    hash_password,
    verify_password,
)
from app.utils.responses import raise_http_error

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")


class TokenPayload(BaseModel):
    sub: str
    role: str
    rags: list[str]
    exp: datetime


def create_access_token(user: Dict[str, Any], expires_in_minutes: int = 60) -> str:
    now = datetime.now(timezone.utc)
    payload = {
        "sub": str(user["_id"]),
        "role": user.get("role", "user"),
        "rags": user.get("rags", []),
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(minutes=expires_in_minutes)).timestamp()),
        "corr_id": get_corr_id(),
    }
    return jwt.encode(payload, settings.jwt_secret, algorithm="HS256")


async def authenticate_user(email: str, password: str) -> Dict[str, Any]:
    user = await find_user_by_email(email)
    if not user or not verify_password(password, user.get("password_hash", "")):
        raise_http_error("INVALID_CREDENTIALS", "Invalid email or password", status.HTTP_401_UNAUTHORIZED)
    return user


async def get_current_user(token: str = Depends(oauth2_scheme)) -> Dict[str, Any]:
    try:
        payload = jwt.decode(token, settings.jwt_secret, algorithms=["HS256"])
        token_data = TokenPayload(
            sub=payload["sub"],
            role=payload["role"],
            rags=payload.get("rags", []),
            exp=datetime.fromtimestamp(payload["exp"], tz=timezone.utc),
        )
    except jwt.PyJWTError:
        raise_http_error("INVALID_TOKEN", "Invalid or expired token", status.HTTP_401_UNAUTHORIZED)

    user = await find_user_by_id(token_data.sub)
    if not user:
        raise_http_error("USER_NOT_FOUND", "User no longer exists", status.HTTP_401_UNAUTHORIZED)
    return user


async def require_admin(user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
    if user.get("role") != "admin":
        raise_http_error("FORBIDDEN", "Admin access required", status.HTTP_403_FORBIDDEN)
    return user

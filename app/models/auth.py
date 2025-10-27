# generated-by: codex-agent 2025-02-15T00:15:00Z
"""
Auth and user models that match the canonical Zod schemas.
"""

from __future__ import annotations

from datetime import datetime
from typing import List, Literal, Optional

from pydantic import BaseModel, Field, ConfigDict

from .common import EmailStr, ObjectIdStr, RagSlug

UserRole = Literal["user", "admin"]


class UserPublic(BaseModel):
    model_config = ConfigDict(
        populate_by_name=True,
        serialize_by_alias=True,
        json_encoders={datetime: lambda dt: dt.isoformat().replace("+00:00", "Z")},
    )

    id: ObjectIdStr = Field(alias="_id")
    email: EmailStr
    name: str
    role: UserRole
    rags: List[RagSlug]
    created_at: datetime
    updated_at: datetime


class RegisterRequest(BaseModel):
    email: EmailStr
    name: str = Field(min_length=1)
    password: str = Field(min_length=8)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8)


class AuthResponse(BaseModel):
    token: str
    user: UserPublic
    corr_id: str


class AdminCreateUserRequest(BaseModel):
    email: EmailStr
    name: str
    role: UserRole
    rags: Optional[List[RagSlug]] = None


class UsersListResponse(BaseModel):
    items: List[UserPublic]
    next_cursor: Optional[str] = None
    corr_id: str

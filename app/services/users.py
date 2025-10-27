# generated-by: codex-agent 2025-02-15T00:19:00Z
"""
User persistence and auth helpers.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorCollection
from passlib.context import CryptContext

from app.models.auth import AdminCreateUserRequest, RegisterRequest, UserPublic, UserRole

pwd_context = CryptContext(schemes=["bcrypt_sha256", "bcrypt"], deprecated="auto")


def get_collection() -> AsyncIOMotorCollection:
    from app.core.database import get_collection as _get_collection

    return _get_collection("users")


async def ensure_indexes() -> None:
    col = get_collection()
    await col.create_index("email", unique=True)
    await col.create_index("rags")


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(password: str, hash_: str) -> bool:
    return pwd_context.verify(password, hash_)


def user_to_public(doc: Dict[str, Any]) -> UserPublic:
    return UserPublic(
        id=str(doc["_id"]),
        email=doc["email"],
        name=doc["name"],
        role=doc.get("role", "user"),
        rags=doc.get("rags", []),
        created_at=doc.get("created_at"),
        updated_at=doc.get("updated_at"),
    )


async def create_user_from_register(payload: RegisterRequest) -> Dict[str, Any]:
    col = get_collection()
    now = datetime.now(timezone.utc)
    doc = {
        "email": payload.email,
        "password_hash": hash_password(payload.password),
        "name": payload.name,
        "role": "user",
        "rags": [],
        "created_at": now,
        "updated_at": now,
    }
    result = await col.insert_one(doc)
    doc["_id"] = result.inserted_id
    return doc


async def create_user_from_admin(payload: AdminCreateUserRequest, generated_password: str) -> Dict[str, Any]:
    col = get_collection()
    now = datetime.now(timezone.utc)
    doc = {
        "email": payload.email,
        "password_hash": hash_password(generated_password),
        "name": payload.name,
        "role": payload.role,
        "rags": payload.rags or [],
        "created_at": now,
        "updated_at": now,
    }
    result = await col.insert_one(doc)
    doc["_id"] = result.inserted_id
    return doc


async def find_user_by_email(email: str) -> Optional[Dict[str, Any]]:
    col = get_collection()
    return await col.find_one({"email": email})


async def find_user_by_id(user_id: str) -> Optional[Dict[str, Any]]:
    col = get_collection()
    return await col.find_one({"_id": ObjectId(user_id)})


async def update_user_rags(user_id: ObjectId, rags: List[str]) -> None:
    col = get_collection()
    await col.update_one({"_id": user_id}, {"$set": {"rags": rags, "updated_at": datetime.now(timezone.utc)}})


async def list_users(limit: int = 50, cursor: Optional[str] = None) -> Tuple[List[UserPublic], Optional[str]]:
    col = get_collection()
    query: Dict[str, Any] = {}
    if cursor:
        query["_id"] = {"$gt": ObjectId(cursor)}
    docs: List[Dict[str, Any]] = []
    async for doc in col.find(query).sort("_id", 1).limit(limit + 1):
        docs.append(doc)

    next_cursor = None
    if len(docs) > limit:
        next_cursor = str(docs[-1]["_id"])
        docs = docs[:-1]

    return [user_to_public(doc) for doc in docs], next_cursor

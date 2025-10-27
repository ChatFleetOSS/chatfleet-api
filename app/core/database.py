# generated-by: codex-agent 2025-02-15T00:12:00Z
"""
MongoDB connection helpers using Motor.
"""

from __future__ import annotations

from typing import Any

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

from .config import settings

_client: AsyncIOMotorClient | None = None


def get_client() -> AsyncIOMotorClient:
    global _client
    if _client is None:
        _client = AsyncIOMotorClient(settings.mongo_uri, uuidRepresentation="standard")
    return _client


def get_database() -> AsyncIOMotorDatabase:
    client = get_client()
    return client.get_database()  # default DB from URI


def get_collection(name: str) -> Any:
    db = get_database()
    return db[name]

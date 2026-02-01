# generated-by: codex-agent 2025-02-15T00:18:00Z
"""
SystemLog persistence helpers.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal, Optional

from bson import ObjectId

from app.core.corr_id import get_corr_id
from app.core.database import get_collection

SystemLogLevel = Literal["info", "warn", "error"]


async def write_system_log(
    event: str,
    rag_slug: Optional[str] = None,
    user_id: Optional[str] = None,
    details: Optional[dict[str, Any]] = None,
    level: SystemLogLevel = "info",
) -> None:
    try:
        user_obj_id = ObjectId(user_id) if user_id else None
    except Exception:
        user_obj_id = None
    doc: dict[str, Any] = {
        "event": event,
        "rag_slug": rag_slug,
        "user_id": user_obj_id,
        "details": details or {},
        "level": level,
        "corr_id": get_corr_id(),
        "timestamp": datetime.now(timezone.utc),
    }
    collection = get_collection("system_logs")
    await collection.insert_one(doc)

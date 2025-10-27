# generated-by: codex-agent 2025-02-15T00:22:00Z
"""
Application bootstrap helpers executed on startup.
"""

from __future__ import annotations

from app.services.rags import ensure_indexes as ensure_rag_indexes
from app.services.users import ensure_indexes as ensure_user_indexes


async def run_startup() -> None:
    await ensure_user_indexes()
    await ensure_rag_indexes()

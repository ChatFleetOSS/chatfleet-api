# generated-by: codex-agent 2025-02-15T00:17:00Z
"""
Admin configuration schemas.
"""

from __future__ import annotations

from pydantic import BaseModel


class AdminConfigResponse(BaseModel):
    chat_model: str
    embed_model: str
    index_dir: str
    upload_dir: str
    max_upload_mb: int
    corr_id: str

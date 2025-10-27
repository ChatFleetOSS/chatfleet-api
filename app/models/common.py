# generated-by: codex-agent 2025-02-15T00:15:00Z
"""
Common types shared across Pydantic models to mirror the canonical Zod schemas.
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated

from pydantic import BaseModel, Field

ObjectIdStr = Annotated[str, Field(pattern=r"^[a-fA-F0-9]{24}$")]
UUIDStr = Annotated[str, Field(pattern=r"^[0-9a-fA-F-]{36}$")]
ISODateStr = Annotated[str, Field(pattern=r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z$")]
EmailStr = Annotated[str, Field(pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$")]
RagSlug = Annotated[str, Field(pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$", min_length=1, max_length=80)]


class TimestampedModel(BaseModel):
    created_at: datetime
    updated_at: datetime

    class Config:
        json_encoders = {datetime: lambda dt: dt.isoformat().replace("+00:00", "Z")}

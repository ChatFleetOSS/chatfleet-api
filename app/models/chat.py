# generated-by: codex-agent 2025-02-15T00:16:00Z
"""
Chat request/response schemas.
"""

from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field

from .common import RagSlug, UUIDStr

MessageRole = Literal["system", "user", "assistant"]


class ChatMessage(BaseModel):
    role: MessageRole
    content: str


class ChatOptions(BaseModel):
    top_k: Optional[int] = Field(default=6, ge=1)
    temperature: Optional[float] = Field(default=0.2, ge=0, le=1)
    max_tokens: Optional[int] = Field(default=500, ge=1)


class ChatRequest(BaseModel):
    rag_slug: RagSlug
    messages: List[ChatMessage] = Field(min_length=1)
    opts: Optional[ChatOptions] = None


class Citation(BaseModel):
    doc_id: UUIDStr
    filename: str
    pages: List[int] = Field(min_length=1)
    snippet: str = Field(max_length=1000)


class Usage(BaseModel):
    tokens_in: int = Field(ge=0)
    tokens_out: int = Field(ge=0)


class ChatResponse(BaseModel):
    answer: str
    citations: List[Citation]
    usage: Usage
    corr_id: str

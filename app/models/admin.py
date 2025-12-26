# generated-by: codex-agent 2025-02-15T00:17:00Z
"""
Admin configuration schemas.
"""

from __future__ import annotations

from pydantic import BaseModel, Field
from typing import Optional, Literal
from datetime import datetime


class AdminConfigResponse(BaseModel):
    chat_model: str
    embed_model: str
    index_dir: str
    upload_dir: str
    max_upload_mb: int
    corr_id: str


LLMProvider = Literal["openai", "vllm"]


class LLMConfigView(BaseModel):
    provider: LLMProvider = Field(default="openai")
    base_url: Optional[str] = None
    chat_model: str
    embed_model: str
    temperature_default: float = 0.2
    top_k_default: int = 6
    has_api_key: bool = False
    verified_at: Optional[datetime] = None
    runtime_enabled: bool = True


class LLMConfigUpdateRequest(BaseModel):
    provider: LLMProvider
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    chat_model: str
    embed_model: str
    temperature_default: Optional[float] = None
    top_k_default: Optional[int] = None


class LLMConfigResponse(BaseModel):
    config: LLMConfigView
    corr_id: str


class LLMConfigTestRequest(BaseModel):
    provider: LLMProvider
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    chat_model: Optional[str] = None
    embed_model: Optional[str] = None


class LLMConfigTestResult(BaseModel):
    ok: bool
    message: Optional[str] = None
    corr_id: str

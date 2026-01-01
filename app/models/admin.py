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
EmbedProvider = Literal["openai", "local"]


class LLMConfigView(BaseModel):
    provider: LLMProvider = Field(default="openai")
    base_url: Optional[str] = None
    chat_model: str
    embed_model: str
    embed_provider: EmbedProvider = Field(default="openai")
    temperature_default: float = 0.2
    top_k_default: int = 6
    index_dir: str
    upload_dir: str
    max_upload_mb: int = 50
    has_api_key: bool = False
    verified_at: Optional[datetime] = None
    runtime_enabled: bool = True


class LLMConfigUpdateRequest(BaseModel):
    provider: LLMProvider
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    chat_model: str
    embed_model: str
    embed_provider: EmbedProvider = Field(default="openai")
    temperature_default: Optional[float] = None
    top_k_default: Optional[int] = None
    index_dir: Optional[str] = None
    upload_dir: Optional[str] = None
    max_upload_mb: Optional[int] = None


class LLMConfigResponse(BaseModel):
    config: LLMConfigView
    corr_id: str


class LLMConfigTestRequest(BaseModel):
    provider: LLMProvider
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    chat_model: Optional[str] = None
    embed_model: Optional[str] = None
    embed_provider: Optional[EmbedProvider] = None


class LLMConfigTestResult(BaseModel):
    ok: bool
    message: Optional[str] = None
    corr_id: str


class LLMModelsRequest(BaseModel):
    provider: LLMProvider
    base_url: Optional[str] = None
    api_key: Optional[str] = None


class LLMModelsResponse(BaseModel):
    chat_models: list[str]
    embed_models: list[str]
    raw_models: list[str]
    corr_id: str


class LLMEmbedTestResult(BaseModel):
    ok: bool
    dim: Optional[int] = None
    message: Optional[str] = None
    corr_id: str

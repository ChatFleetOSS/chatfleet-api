# generated-by: codex-agent 2025-02-15T00:17:00Z
"""
Admin configuration schemas.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator
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
RetrievalMode = Literal["hybrid", "semantic"]


class RetrievalConfig(BaseModel):
    mode: RetrievalMode = "hybrid"
    top_k_default: int = Field(default=12, ge=1, le=200)
    semantic_min_score: float = Field(default=0.2, ge=0.0, le=1.0)
    candidate_multiplier: int = Field(default=4, ge=1, le=10)
    candidate_min: int = Field(default=24, ge=1, le=200)
    rrf_k: int = Field(default=60, ge=1, le=200)
    semantic_weight: float = Field(default=1.0, ge=0.0, le=5.0)
    lexical_weight: float = Field(default=1.0, ge=0.0, le=5.0)
    bm25_k1: float = Field(default=1.5, ge=0.1, le=3.0)
    bm25_b: float = Field(default=0.75, ge=0.0, le=1.0)
    lexical_prewarm: bool = False

    @model_validator(mode="after")
    def validate_weights(self) -> "RetrievalConfig":
        if self.semantic_weight <= 0 and self.lexical_weight <= 0:
            raise ValueError("At least one retrieval weight must be greater than 0")
        return self


class RetrievalConfigUpdate(BaseModel):
    mode: Optional[RetrievalMode] = None
    top_k_default: Optional[int] = Field(default=None, ge=1, le=200)
    semantic_min_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    candidate_multiplier: Optional[int] = Field(default=None, ge=1, le=10)
    candidate_min: Optional[int] = Field(default=None, ge=1, le=200)
    rrf_k: Optional[int] = Field(default=None, ge=1, le=200)
    semantic_weight: Optional[float] = Field(default=None, ge=0.0, le=5.0)
    lexical_weight: Optional[float] = Field(default=None, ge=0.0, le=5.0)
    bm25_k1: Optional[float] = Field(default=None, ge=0.1, le=3.0)
    bm25_b: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    lexical_prewarm: Optional[bool] = None

    @model_validator(mode="after")
    def validate_weights(self) -> "RetrievalConfigUpdate":
        if (
            self.semantic_weight is not None
            and self.lexical_weight is not None
            and self.semantic_weight <= 0
            and self.lexical_weight <= 0
        ):
            raise ValueError("At least one retrieval weight must be greater than 0")
        return self


class LLMConfigView(BaseModel):
    provider: LLMProvider = Field(default="openai")
    base_url: Optional[str] = None
    chat_model: str
    embed_model: str
    embed_provider: EmbedProvider = Field(default="openai")
    temperature_default: float = 0.2
    top_k_default: int = 6
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
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
    retrieval: Optional[RetrievalConfigUpdate] = None
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

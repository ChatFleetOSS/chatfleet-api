# generated-by: codex-agent 2025-02-15T00:15:00Z
"""
Response envelopes with corr_id field (success + error).
"""

from __future__ import annotations

from typing import Generic, TypeVar

from pydantic import BaseModel

from app.core.corr_id import get_corr_id

T = TypeVar("T")


class ErrorDetail(BaseModel):
    code: str
    message: str


class ErrorEnvelope(BaseModel):
    error: ErrorDetail
    corr_id: str

    @classmethod
    def from_error(cls, code: str, message: str) -> "ErrorEnvelope":
        return cls(error=ErrorDetail(code=code, message=message), corr_id=get_corr_id())


class DataEnvelope(BaseModel, Generic[T]):
    data: T
    corr_id: str

    @classmethod
    def wrap(cls, payload: T) -> "DataEnvelope[T]":
        return cls(data=payload, corr_id=get_corr_id())

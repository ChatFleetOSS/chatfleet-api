# generated-by: codex-agent 2025-02-15T00:15:00Z
"""
Response helpers that add `corr_id` fields and shape success/error envelopes.
"""

from __future__ import annotations

from typing import Any, Dict

from fastapi import HTTPException, status
from fastapi.responses import JSONResponse

from app.core.corr_id import get_corr_id
from app.models.envelope import ErrorEnvelope


def with_corr_id(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Add `corr_id` to the payload, returning a new dict."""

    enriched = payload.copy()
    enriched["corr_id"] = get_corr_id()
    return enriched


def raise_http_error(code: str, message: str, status_code: int = status.HTTP_400_BAD_REQUEST) -> None:
    """Raise an HTTPException with the expected ErrorEnvelope body."""

    envelope = ErrorEnvelope.from_error(code=code, message=message)
    raise HTTPException(status_code=status_code, detail=envelope.model_dump())


class UTF8JSONResponse(JSONResponse):
    """JSON response enforcing UTF-8 charset for pact compatibility."""

    media_type = "application/json; charset=utf-8"

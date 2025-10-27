# generated-by: codex-agent 2025-02-15T00:12:00Z
"""
Request-scoped correlation identifiers for observability.
"""

from __future__ import annotations

import time
import uuid
from contextvars import ContextVar
from typing import Awaitable, Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

_corr_id_ctx: ContextVar[str] = ContextVar("corr_id", default="")


def get_corr_id() -> str:
    """Return the current correlation identifier, generating one when absent."""

    corr_id = _corr_id_ctx.get()
    if not corr_id:
        corr_id = str(uuid.uuid4())
        _corr_id_ctx.set(corr_id)
    return corr_id


class CorrIdMiddleware(BaseHTTPMiddleware):
    """Assigns a corr_id to each incoming request and exposes it via headers."""

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        corr_id = request.headers.get("x-corr-id", str(uuid.uuid4()))
        token = _corr_id_ctx.set(corr_id)
        start = time.monotonic()
        try:
            response = await call_next(request)
        finally:
            _corr_id_ctx.reset(token)
        duration_ms = int((time.monotonic() - start) * 1000)
        response.headers["x-corr-id"] = corr_id
        response.headers["x-response-time-ms"] = str(duration_ms)
        return response

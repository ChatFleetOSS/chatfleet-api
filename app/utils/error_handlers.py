# generated-by: codex-agent 2025-02-15T00:29:00Z
"""
Centralised HTTP error handling that emits the canonical ErrorEnvelope.
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.models.envelope import ErrorEnvelope


def install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(HTTPException)
    async def _http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:  # type: ignore[override]
        detail = exc.detail
        if isinstance(detail, dict) and "error" in detail and "corr_id" in detail:
            return JSONResponse(status_code=exc.status_code, content=detail)
        envelope = ErrorEnvelope.from_error(
            code="HTTP_ERROR",
            message=str(detail),
        )
        return JSONResponse(status_code=exc.status_code, content=envelope.model_dump())

    @app.exception_handler(RequestValidationError)
    async def _validation_exception_handler(
        request: Request,
        exc: RequestValidationError,
    ) -> JSONResponse:
        errors = exc.errors()
        message = "; ".join(err.get("msg", "validation error") for err in errors)
        envelope = ErrorEnvelope.from_error(code="VALIDATION_ERROR", message=message)
        return JSONResponse(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, content=envelope.model_dump())

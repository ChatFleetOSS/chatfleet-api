# generated-by: codex-agent 2025-02-15T00:28:00Z
"""
Chat routes (sync + SSE streaming).
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, status
from fastapi.responses import StreamingResponse

from app.dependencies.auth import get_current_user
from app.models.chat import ChatRequest, ChatResponse
from app.services.chat import handle_chat, stream_chat
from app.utils.responses import raise_http_error

router = APIRouter(prefix="/chat", tags=["Chat"])


def _ensure_access(user, rag_slug: str) -> None:
    if rag_slug not in user.get("rags", []) and user.get("role") != "admin":
        raise_http_error("FORBIDDEN", f"User lacks access to rag '{rag_slug}'", status.HTTP_403_FORBIDDEN)


@router.post("", response_model=ChatResponse)
async def chat_completion(payload: ChatRequest, current_user = Depends(get_current_user)) -> ChatResponse:
    _ensure_access(current_user, payload.rag_slug)
    return await handle_chat(payload, str(current_user["_id"]))


@router.post("/stream")
async def chat_stream(payload: ChatRequest, current_user = Depends(get_current_user)) -> StreamingResponse:
    _ensure_access(current_user, payload.rag_slug)
    generator = stream_chat(payload, str(current_user["_id"]))
    return StreamingResponse(generator, media_type="text/event-stream")

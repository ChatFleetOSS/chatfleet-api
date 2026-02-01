# generated-by: codex-agent 2026-02-01T00:00:00Z
"""
Public RAG endpoints (no authentication). Only RAGs marked visibility=public are exposed here.
"""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Query, status
from fastapi.responses import StreamingResponse

from app.models.chat import ChatRequest, ChatResponse
from app.models.rag import RagDocsResponse, RagListResponse
from app.services.chat import handle_chat, stream_chat
from app.services.rags import get_rag_by_slug, list_public_rags
from app.utils.responses import raise_http_error, with_corr_id

router = APIRouter(prefix="/public", tags=["Public RAG"])


async def _get_public_rag_or_404(slug: str) -> dict:
    rag = await get_rag_by_slug(slug)
    if not rag or rag.get("visibility", "private") != "public":
        raise_http_error("RAG_NOT_FOUND", f"Public RAG '{slug}' not found", status.HTTP_404_NOT_FOUND)
    return rag


@router.get("/rag/list", response_model=RagListResponse)
async def public_rag_list(
    limit: int = Query(50, ge=1, le=200),
    cursor: Optional[str] = Query(default=None),
) -> RagListResponse:
    items, next_cursor = await list_public_rags(limit=limit, cursor=cursor)
    payload = {"items": items, "next_cursor": next_cursor}
    return RagListResponse(**with_corr_id(payload))


@router.get("/rag/docs", response_model=RagDocsResponse)
async def public_rag_docs(
    rag_slug: str = Query(...),
) -> RagDocsResponse:
    rag = await _get_public_rag_or_404(rag_slug)
    docs = rag.get("docs", [])
    return RagDocsResponse(
        **with_corr_id(
            {
                "rag_slug": rag_slug,
                "docs": [
                    {
                        "doc_id": entry.get("doc_id"),
                        "filename": entry.get("filename"),
                        "path": entry.get("path"),
                        "mime": entry.get("mime"),
                        "size_bytes": entry.get("size_bytes", 0),
                        "sha256": entry.get("sha256"),
                        "status": entry.get("status", "uploaded"),
                        "chunk_count": entry.get("chunk_count", 0),
                        "error": entry.get("error"),
                        "uploaded_at": entry.get("uploaded_at"),
                        "indexed_at": entry.get("indexed_at"),
                    }
                    for entry in docs
                ],
            }
        )
    )


@router.post("/chat", response_model=ChatResponse)
async def public_chat(payload: ChatRequest) -> ChatResponse:
    await _get_public_rag_or_404(payload.rag_slug)
    return await handle_chat(payload, user_id="public")


@router.post("/chat/stream")
async def public_chat_stream(payload: ChatRequest) -> StreamingResponse:
    await _get_public_rag_or_404(payload.rag_slug)
    generator = stream_chat(payload, user_id="public")
    return StreamingResponse(generator, media_type="text/event-stream")

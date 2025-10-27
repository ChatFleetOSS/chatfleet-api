# generated-by: codex-agent 2025-02-15T00:26:00Z
"""
Admin routes for document uploads, indexing, and status polling.
"""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, File, Form, Query, UploadFile, status
from pydantic import BaseModel

from app.dependencies.auth import require_admin
from app.models.jobs import JobAccepted
from app.models.rag import (
    IndexStatusResponse,
    RagCreateRequest,
    RagCreateResponse,
    RagDeleteRequest,
    RagDeleteResponse,
    RagDocsResponse,
    RagUploadAccepted,
)
from app.services.logging import write_system_log
from app.services.rags import (
    create_rag,
    delete_rag,
    get_docs,
    get_index_status,
    rebuild_index,
    reset_index,
    upload_documents,
)
from app.utils.responses import raise_http_error, with_corr_id

router = APIRouter(tags=["RAG Admin"])


class RagRebuildRequest(BaseModel):
    rag_slug: str


class RagResetRequest(BaseModel):
    rag_slug: str
    confirm: bool


@router.post("/rag", response_model=RagCreateResponse, status_code=status.HTTP_201_CREATED)
async def rag_create(
    payload: RagCreateRequest,
    admin_user = Depends(require_admin),
) -> RagCreateResponse:
    summary = await create_rag(payload, str(admin_user["_id"]))
    return RagCreateResponse(**with_corr_id({"rag": summary}))


@router.post("/rag/upload", response_model=RagUploadAccepted, status_code=status.HTTP_202_ACCEPTED)
async def rag_upload(
    rag_slug: str = Form(...),
    files: list[UploadFile] = File(...),
    splitter_opts: Optional[str] = Form(default=None),
    admin_user = Depends(require_admin),
) -> RagUploadAccepted:
    result = await upload_documents(rag_slug, files, str(admin_user["_id"]))
    await write_system_log(
        event="rag.upload.accepted",
        rag_slug=rag_slug,
        user_id=str(admin_user["_id"]),
        details={"accepted": result.accepted, "skipped": result.skipped},
    )
    return result


@router.get("/rag/docs", response_model=RagDocsResponse)
async def rag_docs(
    rag_slug: str = Query(...),
    admin_user = Depends(require_admin),
) -> RagDocsResponse:
    return await get_docs(rag_slug)


@router.get("/rag/index/status", response_model=IndexStatusResponse)
async def rag_index_status(
    rag_slug: str = Query(...),
    admin_user = Depends(require_admin),
) -> IndexStatusResponse:
    return await get_index_status(rag_slug)


@router.post("/rag/rebuild", response_model=JobAccepted, status_code=status.HTTP_202_ACCEPTED)
async def rag_rebuild(
    payload: RagRebuildRequest,
    admin_user = Depends(require_admin),
) -> JobAccepted:
    job_id = await rebuild_index(payload.rag_slug, str(admin_user["_id"]))
    return JobAccepted(**with_corr_id({"job_id": job_id}))


@router.post("/rag/reset", response_model=JobAccepted, status_code=status.HTTP_202_ACCEPTED)
async def rag_reset(
    payload: RagResetRequest,
    admin_user = Depends(require_admin),
) -> JobAccepted:
    if not payload.confirm:
        raise_http_error("CONFIRMATION_REQUIRED", "Set confirm=true to reset the index", status.HTTP_400_BAD_REQUEST)
    job_id = await reset_index(payload.rag_slug, str(admin_user["_id"]))
    return JobAccepted(**with_corr_id({"job_id": job_id}))


@router.post("/rag/delete", response_model=RagDeleteResponse)
async def rag_delete(
    payload: RagDeleteRequest,
    admin_user = Depends(require_admin),
) -> RagDeleteResponse:
    await delete_rag(payload.rag_slug, payload.confirmation, str(admin_user["_id"]))
    return RagDeleteResponse(**with_corr_id({"deleted": True, "rag_slug": payload.rag_slug}))

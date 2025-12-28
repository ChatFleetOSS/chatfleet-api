# generated-by: codex-agent 2025-02-15T00:20:00Z
"""
RAG persistence, document lifecycle management, and access control helpers.
"""

from __future__ import annotations

import asyncio
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from uuid import uuid4

from bson import ObjectId
from fastapi import UploadFile, status
from motor.motor_asyncio import AsyncIOMotorCollection
from PyPDF2 import PdfReader
from pymongo.errors import DuplicateKeyError

from app.core.config import settings
from app.core.database import get_collection as mongo_collection
from app.models.rag import (
    IndexStatusEnum,
    IndexStatusResponse,
    RagDoc,
    RagDocsResponse,
    RagSummary,
    RagUploadAccepted,
    RagCreateRequest,
    RagUser,
    RagUsersResponse,
)
from app.services.chunker import chunk_text
from app.services.embeddings import embed_texts
from app.services.jobs import JobRecord, job_manager
from app.services.logging import write_system_log
from app.services.users import find_user_by_id, update_user_rags
from app.services.vectorstore import build_index as rebuild_vector_index
from app.services.vectorstore import persist_doc_payload, remove_doc_payload
from app.services.runtime_config import get_runtime_overrides_sync, get_llm_config
from app.utils.responses import raise_http_error, with_corr_id


def get_collection() -> AsyncIOMotorCollection:
    return mongo_collection("rags")


async def ensure_indexes() -> None:
    col = get_collection()
    await col.create_index("slug", unique=True)
    await col.create_index("users")
    await col.create_index("docs.doc_id")


def _doc_to_summary(doc: Dict[str, Any]) -> RagSummary:
    return RagSummary(
        slug=doc["slug"],
        name=doc.get("name", doc["slug"]),
        description=doc.get("description", ""),
        chunks=doc.get("chunks", 0),
        last_updated=doc.get("last_updated", datetime.now(timezone.utc)),
    )


def _doc_to_rag_doc(entry: Dict[str, Any]) -> RagDoc:
    return RagDoc(
        doc_id=entry["doc_id"],
        filename=entry["filename"],
        path=entry.get("path"),
        mime=entry.get("mime", "application/pdf"),
        size_bytes=entry.get("size_bytes", 0),
        sha256=entry.get("sha256"),
        status=entry.get("status", "uploaded"),
        chunk_count=entry.get("chunk_count", 0),
        error=entry.get("error"),
        uploaded_at=entry.get("uploaded_at"),
        indexed_at=entry.get("indexed_at"),
    )


async def _extract_pdf_text(path: str) -> str:
    def _read() -> str:
        reader = PdfReader(path)
        contents = []
        for page in reader.pages:
            try:
                snippet = page.extract_text() or ""
            except Exception:
                snippet = ""
            contents.append(snippet)
        return "\n".join(contents)

    return await asyncio.to_thread(_read)


async def _persist_chunks_and_vectors(
    rag_slug: str,
    doc_entry: Dict[str, Any],
    chunks: Sequence[str],
) -> int:
    if not chunks:
        raise ValueError("No textual chunks extracted")
    embeddings = await embed_texts(chunks)
    # Enforce consistent embedding dimension per RAG
    dim = len(embeddings[0]) if embeddings and embeddings[0] is not None else 0
    try:
        current_rag = await get_rag_by_slug(rag_slug)
    except Exception:
        current_rag = None
    if current_rag:
        # Prefer existing index dim if built
        existing_dim = 0
        idx = current_rag.get("index", {}) if isinstance(current_rag, dict) else {}
        if isinstance(idx, dict):
            existing_dim = int(idx.get("dim") or 0)
        embed_dim_field = int(current_rag.get("embed_dim") or 0)
        guard_dim = existing_dim or embed_dim_field
        if guard_dim and dim and guard_dim != dim:
            raise ValueError(f"EMBED_DIM_MISMATCH: current={dim} existing={guard_dim}")
        # Persist embed_dim on first embed if not set
        if not guard_dim and dim:
            col = get_collection()
            await col.update_one({"slug": rag_slug}, {"$set": {"embed_dim": int(dim)}})
    persist_doc_payload(rag_slug, doc_entry["doc_id"], doc_entry["filename"], chunks, embeddings)
    return len(chunks)


async def _update_rag_index_summary(rag_slug: str, total_chunks: int, dimension: int) -> None:
    col = get_collection()
    await col.update_one(
        {"slug": rag_slug},
        {
            "$set": {
                "chunks": total_chunks,
                "index": {
                    "type": "faiss",
                    "path": str(settings.index_dir / rag_slug / "index.faiss"),
                    # Store runtime embed model name for observability
                    "emb_model": (await get_llm_config()).embed_model if dimension else settings.embed_model,
                    "dim": dimension,
                    "built_at": datetime.now(timezone.utc),
                },
                "last_updated": datetime.now(timezone.utc),
            }
        },
    )


async def list_rags_for_slugs(slugs: List[str], limit: int = 50, cursor: Optional[str] = None) -> Tuple[List[RagSummary], Optional[str]]:
    if not slugs:
        return [], None
    col = get_collection()
    query: Dict[str, Any] = {"slug": {"$in": slugs}}
    if cursor:
        query["_id"] = {"$gt": ObjectId(cursor)}
    docs: List[Dict[str, Any]] = []
    async for doc in col.find(query).sort("_id", 1).limit(limit + 1):
        docs.append(doc)
    next_cursor = None
    if len(docs) > limit:
        next_cursor = str(docs[-1]["_id"])
        docs = docs[:-1]
    return [_doc_to_summary(doc) for doc in docs], next_cursor


async def list_all_rags(limit: int = 50, cursor: Optional[str] = None) -> Tuple[List[RagSummary], Optional[str]]:
    col = get_collection()
    query: Dict[str, Any] = {}
    if cursor:
        query["_id"] = {"$gt": ObjectId(cursor)}
    docs: List[Dict[str, Any]] = []
    async for doc in col.find(query).sort("_id", 1).limit(limit + 1):
        docs.append(doc)
    next_cursor = None
    if len(docs) > limit:
        next_cursor = str(docs[-1]["_id"])
        docs = docs[:-1]
    return [_doc_to_summary(doc) for doc in docs], next_cursor


async def get_rag_by_slug(slug: str) -> Optional[Dict[str, Any]]:
    col = get_collection()
    return await col.find_one({"slug": slug})


async def create_rag(payload: RagCreateRequest, creator_id: str) -> RagSummary:
    col = get_collection()
    now = datetime.now(timezone.utc)
    doc: Dict[str, Any] = {
        "slug": payload.slug,
        "name": payload.name,
        "description": payload.description,
        "users": [],
        "docs": [],
        "chunks": 0,
        "index": {
            "type": "faiss",
            "path": "",
            "emb_model": settings.embed_model,
            "dim": 0,
            "built_at": None,
        },
        "created_at": now,
        "last_updated": now,
        "created_by": ObjectId(creator_id),
    }
    try:
        result = await col.insert_one(doc)
    except DuplicateKeyError:
        raise_http_error("RAG_EXISTS", f"RAG '{payload.slug}' already exists", status.HTTP_409_CONFLICT)
    doc["_id"] = result.inserted_id
    await write_system_log(
        event="rag.create",
        rag_slug=payload.slug,
        user_id=creator_id,
        details={"name": payload.name},
    )
    return _doc_to_summary(doc)


async def get_docs(slug: str) -> RagDocsResponse:
    rag = await get_rag_by_slug(slug)
    if not rag:
        raise_http_error("RAG_NOT_FOUND", f"RAG '{slug}' not found", status_code=404)
    docs = [_doc_to_rag_doc(entry) for entry in rag.get("docs", [])]
    return RagDocsResponse(**with_corr_id({"rag_slug": slug, "docs": [doc.model_dump() for doc in docs]}))


async def get_index_status(slug: str) -> IndexStatusResponse:
    rag = await get_rag_by_slug(slug)
    if not rag:
        raise_http_error("RAG_NOT_FOUND", f"RAG '{slug}' not found", status_code=404)
    docs = rag.get("docs", [])
    total_docs = len(docs)
    done_docs = sum(1 for doc in docs if doc.get("status") == "indexed")
    total_chunks = sum(doc.get("chunk_count", 0) for doc in docs)
    done_chunks = total_chunks if done_docs == total_docs else sum(
        doc.get("chunk_count", 0) for doc in docs if doc.get("status") == "indexed"
    )
    status: IndexStatusEnum = "idle"
    if any(doc.get("status") in {"chunking", "indexing"} for doc in docs):
        status = "building"
    if any(doc.get("status") == "error" for doc in docs):
        status = "error"
    progress = 0.0 if total_docs == 0 else done_docs / total_docs
    index_info = rag.get("index", {})
    error = index_info.get("error")
    payload = {
        "rag_slug": slug,
        "status": status,
        "progress": progress,
        "total_docs": total_docs,
        "done_docs": done_docs,
        "total_chunks": total_chunks,
        "done_chunks": done_chunks,
        "error": error,
    }
    return IndexStatusResponse(**with_corr_id(payload))


def _save_upload_target(rag_slug: str) -> Path:
    _, upload_dir, _, _, _ = get_runtime_overrides_sync()
    target_dir = upload_dir / rag_slug
    target_dir.mkdir(parents=True, exist_ok=True)
    return target_dir


async def _persist_file(rag_slug: str, upload: UploadFile, doc_id: str) -> Tuple[str, int, str]:
    """Persist an uploaded PDF to the rag's upload directory safely.

    - Store using a generated filename `<doc_id>.pdf` to avoid path traversal.
    - Validate basic PDF magic header ("%PDF") before accepting.
    - Enforce max size during streaming to avoid excessive disk usage.
    """

    target_dir = _save_upload_target(rag_slug)
    safe_filename = f"{doc_id}.pdf"
    target_path = target_dir / safe_filename
    size = 0
    hasher = hashlib.sha256()
    max_bytes = settings.max_upload_mb * 1024 * 1024

    first_chunk = await upload.read(8 * 1024)
    if not first_chunk:
        await upload.close()
        raise ValueError("Empty upload")
    # Basic PDF signature check
    if not first_chunk.startswith(b"%PDF"):
        await upload.close()
        raise ValueError("Only PDF files are accepted")

    with target_path.open("wb") as out_file:
        size += len(first_chunk)
        if size > max_bytes:
            await upload.close()
            out_file.close()
            try:
                target_path.unlink()
            except FileNotFoundError:
                pass
            raise ValueError(f"{upload.filename or safe_filename} exceeds {settings.max_upload_mb}MB limit")
        hasher.update(first_chunk)
        out_file.write(first_chunk)

        while True:
            chunk = await upload.read(1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            if size > max_bytes:
                await upload.close()
                out_file.close()
                try:
                    target_path.unlink()
                except FileNotFoundError:
                    pass
                raise ValueError(f"{upload.filename or safe_filename} exceeds {settings.max_upload_mb}MB limit")
            hasher.update(chunk)
            out_file.write(chunk)
    await upload.close()
    return str(target_path), size, hasher.hexdigest()


async def _register_doc_metadata(
    rag: Dict[str, Any],
    filename: str,
    stored_path: str,
    size_bytes: int,
    sha256: str,
) -> Dict[str, Any]:
    doc_id = str(uuid4())
    entry = {
        "doc_id": doc_id,
        "filename": filename,
        "path": stored_path,
        "mime": "application/pdf",
        "size_bytes": size_bytes,
        "sha256": sha256,
        "status": "uploaded",
        "chunk_count": 0,
        "error": None,
        "uploaded_at": datetime.now(timezone.utc),
        "indexed_at": None,
    }
    docs = rag.get("docs", [])
    docs.append(entry)
    rag["docs"] = docs
    rag["last_updated"] = datetime.now(timezone.utc)
    col = get_collection()
    await col.update_one({"_id": rag["_id"]}, {"$set": {"docs": docs, "last_updated": rag["last_updated"]}})
    return entry


async def upload_documents(
    rag_slug: str,
    uploads: List[UploadFile],
    uploader_id: str,
) -> RagUploadAccepted:
    rag = await get_rag_by_slug(rag_slug)
    if not rag:
        raise_http_error("RAG_NOT_FOUND", f"RAG '{rag_slug}' not found", status_code=404)

    accepted: List[str] = []
    skipped: List[str] = []
    new_entries: List[Dict[str, Any]] = []

    for upload in uploads:
        if not (upload.filename or "").lower().endswith(".pdf"):
            skipped.append(upload.filename or "unnamed")
            continue
        try:
            stored_path, size_bytes, sha256 = await _persist_file(rag_slug, upload, str(uuid4()))
        except ValueError as exc:
            skipped.append(upload.filename or "unnamed")
            await write_system_log(
                event="rag.upload.skipped",
                rag_slug=rag_slug,
                user_id=uploader_id,
                details={"reason": str(exc)},
                level="warn",
            )
            continue
        entry = await _register_doc_metadata(
            rag,
            upload.filename or Path(stored_path).name,
            stored_path,
            size_bytes,
            sha256,
        )
        new_entries.append(entry)
        accepted.append(entry["filename"])

    async def runner(job: JobRecord) -> None:
        chunk_counts: Dict[str, int] = {}
        try:
            if not new_entries:
                await write_system_log(
                    event="rag.upload.noop",
                    rag_slug=rag_slug,
                    user_id=uploader_id,
                    details={"job_id": job.job_id, "skipped": skipped},
                    level="warn",
                )
                job.result = {"total_chunks": 0, "dimension": 0}
                return
            for entry in new_entries:
                doc_id = entry["doc_id"]
                try:
                    await _update_doc_status(rag_slug, doc_id, "chunking")

                    text = await _extract_pdf_text(entry["path"])
                    chunks = chunk_text(text)
                    chunk_count = len(chunks)
                    chunk_counts[doc_id] = chunk_count
                    await _update_doc_status(rag_slug, doc_id, "chunked", chunk_count=chunk_count)

                    stored_count = await _persist_chunks_and_vectors(rag_slug, entry, chunks)
                    await _update_doc_status(rag_slug, doc_id, "indexing", chunk_count=stored_count)
                except Exception as doc_exc:
                    await _update_doc_status(rag_slug, doc_id, "error", error=str(doc_exc))
                    raise

            total_chunks, dimension = await asyncio.to_thread(rebuild_vector_index, rag_slug)
            for entry in new_entries:
                doc_id = entry["doc_id"]
                await _update_doc_status(rag_slug, doc_id, "indexed", chunk_count=chunk_counts.get(doc_id, 0))

            await _update_rag_index_summary(rag_slug, total_chunks, dimension)
            await write_system_log(
                event="rag.upload.complete",
                rag_slug=rag_slug,
                user_id=uploader_id,
                details={"accepted": accepted, "skipped": skipped, "job_id": job.job_id, "chunks": total_chunks},
            )
            job.result = {"total_chunks": total_chunks, "dimension": dimension}
        except Exception as exc:  # pragma: no cover — safety net
            job.status = "error"
            job.error = str(exc)
            await write_system_log(
                event="rag.upload.failed",
                rag_slug=rag_slug,
                user_id=uploader_id,
                details={"error": str(exc), "job_id": job.job_id},
                level="error",
            )
            raise

    job = job_manager.schedule("RAG_INDEX", runner)
    payload = {
        "job_id": job.job_id,
        "accepted": accepted,
        "skipped": skipped,
        "rag_slug": rag_slug,
    }
    return RagUploadAccepted(**with_corr_id(payload))


async def _update_doc_status(
    rag_slug: str,
    doc_id: str,
    status: str,
    chunk_count: Optional[int] = None,
    error: Optional[str] = None,
) -> None:
    col = get_collection()
    await col.update_one(
        {"slug": rag_slug, "docs.doc_id": doc_id},
        {
            "$set": {
                "docs.$.status": status,
                "docs.$.chunk_count": chunk_count if chunk_count is not None else 0,
                "docs.$.indexed_at": datetime.now(timezone.utc) if status == "indexed" else None,
                "docs.$.error": error,
                "last_updated": datetime.now(timezone.utc),
            }
        },
    )


async def rebuild_index(rag_slug: str, user_id: str) -> str:
    rag = await get_rag_by_slug(rag_slug)
    if not rag:
        raise_http_error("RAG_NOT_FOUND", f"RAG '{rag_slug}' not found", status_code=404)

    async def runner(job: JobRecord) -> None:
        try:
            current = await get_rag_by_slug(rag_slug)
            docs = current.get("docs", []) if current else []
            for doc in docs:
                if doc.get("status") != "error":
                    await _update_doc_status(
                        rag_slug,
                        doc["doc_id"],
                        "indexing",
                        chunk_count=doc.get("chunk_count", 0),
                    )

            total_chunks, dimension = await asyncio.to_thread(rebuild_vector_index, rag_slug)

            current = await get_rag_by_slug(rag_slug)
            docs = current.get("docs", []) if current else []
            for doc in docs:
                if doc.get("status") != "error":
                    await _update_doc_status(
                        rag_slug,
                        doc["doc_id"],
                        "indexed",
                        chunk_count=doc.get("chunk_count", 0),
                    )

            await _update_rag_index_summary(rag_slug, total_chunks, dimension)
            await write_system_log(
                event="rag.rebuild.complete",
                rag_slug=rag_slug,
                user_id=user_id,
                details={"job_id": job.job_id, "chunks": total_chunks},
            )
            job.result = {"total_chunks": total_chunks, "dimension": dimension}
        except Exception as exc:  # pragma: no cover — defensive
            job.status = "error"
            job.error = str(exc)
            await write_system_log(
                event="rag.rebuild.failed",
                rag_slug=rag_slug,
                user_id=user_id,
                details={"error": str(exc), "job_id": job.job_id},
                level="error",
            )
            raise

    job = job_manager.schedule("RAG_REBUILD", runner)
    return job.job_id


async def reset_index(rag_slug: str, user_id: str) -> str:
    rag = await get_rag_by_slug(rag_slug)
    if not rag:
        raise_http_error("RAG_NOT_FOUND", f"RAG '{rag_slug}' not found", status_code=404)

    async def runner(job: JobRecord) -> None:
        col = get_collection()
        await col.update_one(
            {"_id": rag["_id"]},
            {
                "$set": {
                    "docs": [],
                    "chunks": 0,
                    "index": {"type": "faiss", "path": "", "emb_model": settings.embed_model, "dim": 1536, "built_at": None},
                    "last_updated": datetime.now(timezone.utc),
                }
            },
        )

        for doc in rag.get("docs", []):
            remove_doc_payload(rag_slug, doc["doc_id"])

        index_dir, upload_dir_base, _, _, _ = get_runtime_overrides_sync()
        index_dir = index_dir / rag_slug
        if index_dir.exists():
            for child in index_dir.glob("*"):
                try:
                    if child.is_file():
                        child.unlink()
                    elif child.is_dir():
                        for nested in child.rglob("*"):
                            if nested.is_file():
                                nested.unlink()
                        child.rmdir()
                except OSError:
                    pass

        target_dir = upload_dir_base / rag_slug
        if target_dir.exists():
            for child in target_dir.glob("*"):
                try:
                    if child.is_file():
                        child.unlink()
                    elif child.is_dir():
                        for nested in child.rglob("*"):
                            if nested.is_file():
                                nested.unlink()
                        child.rmdir()
                except OSError:
                    pass

        job.result = {"total_chunks": 0}
        await write_system_log(
            event="rag.reset.complete",
            rag_slug=rag_slug,
            user_id=user_id,
            details={"job_id": job.job_id},
        )

    job = job_manager.schedule("RAG_RESET", runner)
    return job.job_id


async def list_rag_users(rag_slug: str) -> RagUsersResponse:
    rag = await get_rag_by_slug(rag_slug)
    if not rag:
        raise_http_error("RAG_NOT_FOUND", f"RAG '{rag_slug}' not found", status_code=404)
    user_ids: List[ObjectId] = rag.get("users", [])
    users: List[RagUser] = []
    for user_id in user_ids:
        doc = await find_user_by_id(str(user_id))
        if doc:
            users.append(RagUser(id=str(doc["_id"]), email=doc["email"], name=doc["name"], role=doc.get("role", "user")))
    payload = {"rag_slug": rag_slug, "users": [user.model_dump(by_alias=True) for user in users]}
    return RagUsersResponse(**with_corr_id(payload))


async def add_user_access(rag_slug: str, user_id: ObjectId) -> None:
    col = get_collection()
    await col.update_one({"slug": rag_slug}, {"$addToSet": {"users": user_id}})


async def remove_user_access(rag_slug: str, user_id: ObjectId) -> None:
    col = get_collection()
    await col.update_one({"slug": rag_slug}, {"$pull": {"users": user_id}})


async def add_user_to_rag(rag_slug: str, user_id: ObjectId, current_rags: List[str]) -> None:
    current = set(current_rags)
    current.add(rag_slug)
    await update_user_rags(user_id, sorted(current))
    await add_user_access(rag_slug, user_id)


async def remove_user_from_rag(rag_slug: str, user_id: ObjectId, current_rags: List[str]) -> None:
    current = [slug for slug in current_rags if slug != rag_slug]
    await update_user_rags(user_id, current)
    await remove_user_access(rag_slug, user_id)


async def delete_rag(rag_slug: str, confirmation: str, actor_id: str) -> None:
    if confirmation != rag_slug:
        raise_http_error("CONFIRMATION_MISMATCH", "Confirmation must match rag_slug", status.HTTP_422_UNPROCESSABLE_ENTITY)

    rag = await get_rag_by_slug(rag_slug)
    if not rag:
        raise_http_error("RAG_NOT_FOUND", f"RAG '{rag_slug}' not found", status.HTTP_404_NOT_FOUND)

    # Optionally check for lock or concurrent operations here (placeholder for future state).

    # Remove document payloads and indexes.
    for doc in rag.get("docs", []):
        remove_doc_payload(rag_slug, doc["doc_id"])

    index_dir, upload_dir_base, _, _, _ = get_runtime_overrides_sync()
    index_dir = index_dir / rag_slug
    if index_dir.exists():
        for child in index_dir.glob("*"):
            try:
                if child.is_file():
                    child.unlink()
                elif child.is_dir():
                    for nested in child.rglob("*"):
                        if nested.is_file():
                            nested.unlink()
                    child.rmdir()
            except OSError:
                pass
        try:
            index_dir.rmdir()
        except OSError:
            pass

    upload_dir = upload_dir_base / rag_slug
    if upload_dir.exists():
        for child in upload_dir.glob("*"):
            try:
                if child.is_file():
                    child.unlink()
                elif child.is_dir():
                    for nested in child.rglob("*"):
                        if nested.is_file():
                            nested.unlink()
                    child.rmdir()
            except OSError:
                pass
        try:
            upload_dir.rmdir()
        except OSError:
            pass

    # Remove rag slug from users.
    user_ids: List[ObjectId] = rag.get("users", [])
    for user_id in user_ids:
        doc = await find_user_by_id(str(user_id))
        if not doc:
            continue
        current_rags = [slug for slug in doc.get("rags", []) if slug != rag_slug]
        await update_user_rags(user_id, sorted(current_rags))

    # Remove rag entry and access list.
    col = get_collection()
    result = await col.delete_one({"_id": rag["_id"]})
    if result.deleted_count != 1:
        raise_http_error("DELETE_FAILED", f"Unable to delete RAG '{rag_slug}'", status.HTTP_500_INTERNAL_SERVER_ERROR)

    await write_system_log(
        event="rag.delete.complete",
        rag_slug=rag_slug,
        user_id=actor_id,
        details={"docs_deleted": len(rag.get("docs", []))},
    )

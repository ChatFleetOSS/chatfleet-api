# generated-by: codex-agent 2025-02-15T00:46:00Z
"""
FAISS-based vector store per RAG slug.
"""

from __future__ import annotations

import json
import logging
import pickle
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Optional, Sequence

import faiss
import numpy as np

from app.services.runtime_config import get_runtime_overrides_sync

logger = logging.getLogger("chatfleet.vectorstore")
logger.setLevel(logging.INFO)

DOC_BUCKET = "docs"
INDEX_FILE = "index.faiss"
METADATA_FILE = "metadata.json"


@dataclass
class ChunkRecord:
    doc_id: str
    filename: str
    chunk_index: int
    text: str
    page_start: Optional[int] = None
    page_end: Optional[int] = None


def _rag_base_dir(rag_slug: str) -> Path:
    index_dir, _, _, _, _ = get_runtime_overrides_sync()
    base = index_dir / rag_slug
    base.mkdir(parents=True, exist_ok=True)
    (base / DOC_BUCKET).mkdir(parents=True, exist_ok=True)
    return base


def persist_doc_payload(
    rag_slug: str,
    doc_id: str,
    filename: str,
    chunks: Sequence[Any],
    embeddings: Sequence[Sequence[float]],
) -> Path:
    """
    Store chunk texts + embeddings for a single document.
    """

    base = _rag_base_dir(rag_slug)
    doc_path = base / DOC_BUCKET / f"{doc_id}.pkl"
    serialized_chunks: list[dict[str, Any]] = []
    for chunk in chunks:
        # Accept both legacy strings and structured ChunkWithPage payloads.
        if isinstance(chunk, dict):
            text = chunk.get("text", "")
            page_start = chunk.get("page_start")
            page_end = chunk.get("page_end")
        elif hasattr(chunk, "text"):
            text = getattr(chunk, "text")
            page_start = getattr(chunk, "page_start", None)
            page_end = getattr(chunk, "page_end", None)
        else:
            text = str(chunk)
            page_start = None
            page_end = None
        serialized_chunks.append(
            {"text": text, "page_start": page_start, "page_end": page_end}
        )

    payload = {
        "doc_id": doc_id,
        "filename": filename,
        "chunks": serialized_chunks,
        "embeddings": [list(vec) for vec in embeddings],
    }
    with doc_path.open("wb") as handle:
        pickle.dump(payload, handle)
    return doc_path


def remove_doc_payload(rag_slug: str, doc_id: str) -> None:
    path = _rag_base_dir(rag_slug) / DOC_BUCKET / f"{doc_id}.pkl"
    if path.exists():
        path.unlink()


def _load_doc_payload(path: Path) -> dict:
    with path.open("rb") as handle:
        return pickle.load(handle)


def build_index(rag_slug: str) -> tuple[int, int]:
    """
    Build the FAISS index from all persisted doc payloads.

    Returns (total_chunks, dimension). When no documents exist the index
    file is removed and (0, 0) is returned.
    """

    base = _rag_base_dir(rag_slug)
    doc_dir = base / DOC_BUCKET
    vectors: List[np.ndarray] = []
    metadata: List[ChunkRecord] = []

    dims: set[int] = set()

    def _coerce_chunk(entry: Any) -> tuple[str, Optional[int], Optional[int]]:
        if isinstance(entry, dict):
            return (
                entry.get("text", ""),
                entry.get("page_start"),
                entry.get("page_end"),
            )
        if hasattr(entry, "text"):
            return (
                getattr(entry, "text"),
                getattr(entry, "page_start", None),
                getattr(entry, "page_end", None),
            )
        return str(entry), None, None

    for doc_file in sorted(doc_dir.glob("*.pkl")):
        payload = _load_doc_payload(doc_file)
        embeddings = list(payload.get("embeddings", []))
        chunks = list(payload.get("chunks", []))
        filename = payload.get("filename", "document")
        if len(embeddings) != len(chunks):
            raise ValueError(
                "EMBED_PAYLOAD_MISMATCH: "
                f"embeddings={len(embeddings)} chunks={len(chunks)} doc={doc_file.name}"
            )
        for idx, embedding in enumerate(embeddings):
            arr = np.asarray(embedding, dtype=np.float32)
            vectors.append(arr)
            if arr.ndim == 1:
                dims.add(int(arr.shape[0]))
            text, page_start, page_end = _coerce_chunk(
                chunks[idx] if idx < len(chunks) else {}
            )
            metadata.append(
                ChunkRecord(
                    doc_id=payload["doc_id"],
                    filename=filename,
                    chunk_index=idx,
                    text=text,
                    page_start=page_start,
                    page_end=page_end,
                )
            )

    index_path = base / INDEX_FILE
    metadata_path = base / METADATA_FILE

    if not vectors:
        if index_path.exists():
            index_path.unlink()
        if metadata_path.exists():
            metadata_path.unlink()
        return 0, 0

    # Enforce homogeneous dimensions across all vectors
    if len(dims) > 1:
        raise ValueError(
            f"EMBED_DIM_MISMATCH: multiple embedding dimensions detected: {sorted(dims)}"
        )

    matrix = np.stack(vectors).astype("float32")
    faiss.normalize_L2(matrix)
    dim = matrix.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(matrix)
    faiss.write_index(index, str(index_path))

    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(
            [record.__dict__ for record in metadata],
            handle,
            ensure_ascii=False,
        )

    logger.info(
        "rag.index.build rag=%s vectors=%s dim=%s docs=%s",
        rag_slug,
        len(metadata),
        dim,
        len(list(doc_dir.glob("*.pkl"))),
        extra={
            "rag_slug": rag_slug,
            "vectors": len(metadata),
            "dim": dim,
            "docs": len(list(doc_dir.glob("*.pkl"))),
        },
    )
    return len(metadata), dim


def load_index(rag_slug: str) -> tuple[faiss.IndexFlatIP, List[ChunkRecord]]:
    base = _rag_base_dir(rag_slug)
    index_path = base / INDEX_FILE
    metadata_path = base / METADATA_FILE
    if not index_path.exists() or not metadata_path.exists():
        raise FileNotFoundError("Index not built for rag")
    index = faiss.read_index(str(index_path))
    with metadata_path.open("r", encoding="utf-8") as handle:
        entries = json.load(handle)
    records = [
        ChunkRecord(
            doc_id=item["doc_id"],
            filename=item["filename"],
            chunk_index=item["chunk_index"],
            text=item.get("text", ""),
            page_start=item.get("page_start"),
            page_end=item.get("page_end"),
        )
        for item in entries
    ]
    return index, records


def query_index(
    rag_slug: str,
    query_vector: Sequence[float],
    top_k: int,
    min_score: Optional[float] = 0.2,
) -> List[tuple[float, ChunkRecord]]:
    index, metadata = load_index(rag_slug)
    vector = np.asarray(query_vector, dtype="float32")[None, :]
    faiss.normalize_L2(vector)
    k = min(max(top_k, 1), index.ntotal)
    if k == 0:
        return []
    distances, indices = index.search(vector, k)
    raw_scores = distances[0].tolist() if hasattr(distances[0], "tolist") else list(distances[0])
    hits: List[tuple[float, ChunkRecord]] = []
    for score, idx in zip(distances[0], indices[0]):
        if idx < 0 or idx >= len(metadata):
            continue
        if min_score is not None and score < min_score:
            continue
        hits.append((float(score), metadata[idx]))
    logger.info(
        "rag.index.query rag=%s top_k=%s eff_k=%s min_score=%s returned=%s scores=%s",
        rag_slug,
        top_k,
        k,
        min_score,
        len(hits),
        [float(s) for s in raw_scores[:10]],
        extra={
            "rag_slug": rag_slug,
            "requested_top_k": top_k,
            "effective_top_k": k,
            "min_score": min_score,
            "returned": len(hits),
            "scores": [float(s) for s in raw_scores[:10]],
        },
    )
    return hits

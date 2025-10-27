# generated-by: codex-agent 2025-02-15T00:46:00Z
"""
FAISS-based vector store per RAG slug.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Sequence

import faiss
import numpy as np

from app.core.config import settings

DOC_BUCKET = "docs"
INDEX_FILE = "index.faiss"
METADATA_FILE = "metadata.json"


@dataclass
class ChunkRecord:
    doc_id: str
    filename: str
    chunk_index: int
    text: str


def _rag_base_dir(rag_slug: str) -> Path:
    base = settings.index_dir / rag_slug
    base.mkdir(parents=True, exist_ok=True)
    (base / DOC_BUCKET).mkdir(parents=True, exist_ok=True)
    return base


def persist_doc_payload(
    rag_slug: str,
    doc_id: str,
    filename: str,
    chunks: Sequence[str],
    embeddings: Sequence[Sequence[float]],
) -> Path:
    """
    Store chunk texts + embeddings for a single document.
    """

    base = _rag_base_dir(rag_slug)
    doc_path = base / DOC_BUCKET / f"{doc_id}.pkl"
    payload = {
        "doc_id": doc_id,
        "filename": filename,
        "chunks": list(chunks),
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

    for doc_file in sorted(doc_dir.glob("*.pkl")):
        payload = _load_doc_payload(doc_file)
        embeddings = payload.get("embeddings", [])
        chunks = payload.get("chunks", [])
        filename = payload.get("filename", "document")
        for idx, embedding in enumerate(embeddings):
            vectors.append(np.asarray(embedding, dtype=np.float32))
            metadata.append(
                ChunkRecord(
                    doc_id=payload["doc_id"],
                    filename=filename,
                    chunk_index=idx,
                    text=chunks[idx] if idx < len(chunks) else "",
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

    matrix = np.stack(vectors).astype("float32")
    faiss.normalize_L2(matrix)
    dim = matrix.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(matrix)
    faiss.write_index(index, str(index_path))

    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump([record.__dict__ for record in metadata], handle, ensure_ascii=False)

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
        )
        for item in entries
    ]
    return index, records


def query_index(
    rag_slug: str,
    query_vector: Sequence[float],
    top_k: int,
) -> List[tuple[float, ChunkRecord]]:
    index, metadata = load_index(rag_slug)
    vector = np.asarray(query_vector, dtype="float32")[None, :]
    faiss.normalize_L2(vector)
    distances, indices = index.search(vector, top_k)
    hits: List[tuple[float, ChunkRecord]] = []
    for score, idx in zip(distances[0], indices[0]):
        if idx < 0 or idx >= len(metadata):
            continue
        hits.append((float(score), metadata[idx]))
    return hits

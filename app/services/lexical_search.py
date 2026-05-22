# generated-by: codex-agent 2026-05-05T00:00:00Z
"""
Local lexical search over RAG chunk metadata.

This module intentionally depends only on the existing metadata.json produced
by the FAISS index builder. It adds no service, migration, or required artifact.
"""

from __future__ import annotations

import json
import logging
import math
import re
import threading
import time
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from app.services.vectorstore import ChunkRecord, METADATA_FILE, _rag_base_dir

logger = logging.getLogger("chatfleet.lexical_search")
logger.setLevel(logging.INFO)

BM25_K1 = 1.5
BM25_B = 0.75
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[._/-][a-z0-9]+)+|[a-z0-9]{2,}")


@dataclass(frozen=True)
class _CacheSignature:
    path: Path
    mtime_ns: int
    size: int


@dataclass
class LexicalIndex:
    records: list[ChunkRecord]
    term_frequencies: list[Counter[str]]
    document_frequencies: dict[str, int]
    doc_lengths: list[int]
    avg_doc_length: float


_CACHE: dict[str, tuple[_CacheSignature, LexicalIndex]] = {}
_CACHE_LOCK = threading.Lock()


def _normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    decomposed = unicodedata.normalize("NFKD", normalized)
    return "".join(char for char in decomposed if not unicodedata.combining(char))


def tokenize_text(text: str) -> list[str]:
    """Tokenize text for lexical retrieval, preserving useful references."""

    normalized = _normalize_text(text)
    tokens: list[str] = []
    for match in _TOKEN_RE.finditer(normalized):
        token = match.group(0).strip("._/-")
        if not token:
            continue
        tokens.append(token)
        if any(sep in token for sep in ("-", "_", ".", "/")):
            for part in re.split(r"[._/-]+", token):
                if len(part) >= 2 or part.isdigit():
                    tokens.append(part)
    return tokens


def _load_metadata_records(metadata_path: Path) -> list[ChunkRecord]:
    with metadata_path.open("r", encoding="utf-8") as handle:
        entries = json.load(handle)
    return [
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


def _metadata_signature(rag_slug: str) -> _CacheSignature:
    metadata_path = _rag_base_dir(rag_slug) / METADATA_FILE
    stat = metadata_path.stat()
    return _CacheSignature(
        path=metadata_path,
        mtime_ns=stat.st_mtime_ns,
        size=stat.st_size,
    )


def build_lexical_index(records: Sequence[ChunkRecord]) -> LexicalIndex:
    """Build an in-memory BM25 index from chunk records."""

    term_frequencies: list[Counter[str]] = []
    document_frequencies: Counter[str] = Counter()
    doc_lengths: list[int] = []

    for record in records:
        tokens = tokenize_text(record.text)
        frequencies = Counter(tokens)
        term_frequencies.append(frequencies)
        doc_lengths.append(sum(frequencies.values()))
        document_frequencies.update(frequencies.keys())

    avg_doc_length = (
        sum(doc_lengths) / len(doc_lengths) if doc_lengths else 0.0
    )
    return LexicalIndex(
        records=list(records),
        term_frequencies=term_frequencies,
        document_frequencies=dict(document_frequencies),
        doc_lengths=doc_lengths,
        avg_doc_length=avg_doc_length,
    )


def load_lexical_index(rag_slug: str) -> LexicalIndex:
    """Load or reuse the BM25 index for a RAG, invalidating on metadata change."""

    signature = _metadata_signature(rag_slug)
    with _CACHE_LOCK:
        cached = _CACHE.get(rag_slug)
        if cached and cached[0] == signature:
            return cached[1]

    records = _load_metadata_records(signature.path)
    index = build_lexical_index(records)
    with _CACHE_LOCK:
        _CACHE[rag_slug] = (signature, index)
    return index


def prewarm_lexical_index(rag_slug: str) -> LexicalIndex:
    """Build and cache the lexical index eagerly after a RAG index rebuild."""

    start = time.perf_counter()
    index = load_lexical_index(rag_slug)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    logger.info(
        "rag.lexical.prewarm rag=%s chunks=%s elapsed_ms=%.3f",
        rag_slug,
        len(index.records),
        elapsed_ms,
        extra={
            "rag_slug": rag_slug,
            "chunks": len(index.records),
            "elapsed_ms": round(elapsed_ms, 3),
        },
    )
    return index


def _bm25_score(
    query_terms: Iterable[str],
    index: LexicalIndex,
    doc_idx: int,
    *,
    k1: float = BM25_K1,
    b: float = BM25_B,
) -> float:
    if not index.records or index.avg_doc_length <= 0:
        return 0.0

    score = 0.0
    frequencies = index.term_frequencies[doc_idx]
    doc_length = index.doc_lengths[doc_idx]
    total_docs = len(index.records)

    for term in query_terms:
        term_frequency = frequencies.get(term, 0)
        if term_frequency <= 0:
            continue
        document_frequency = index.document_frequencies.get(term, 0)
        if document_frequency <= 0:
            continue
        idf = math.log(
            1.0 + (total_docs - document_frequency + 0.5) / (document_frequency + 0.5)
        )
        length_norm = 1.0 - b + b * (doc_length / index.avg_doc_length)
        numerator = term_frequency * (k1 + 1.0)
        denominator = term_frequency + k1 * length_norm
        score += idf * numerator / denominator
    return score


def search_lexical_index(
    rag_slug: str,
    query: str,
    top_k: int,
    *,
    bm25_k1: float = BM25_K1,
    bm25_b: float = BM25_B,
) -> list[tuple[float, ChunkRecord]]:
    """Return BM25 hits for a query. Only positive BM25 scores are retained."""

    query_terms = tokenize_text(query)
    if not query_terms:
        return []

    index = load_lexical_index(rag_slug)
    scored: list[tuple[float, int, ChunkRecord]] = []
    for doc_idx, record in enumerate(index.records):
        score = _bm25_score(query_terms, index, doc_idx, k1=bm25_k1, b=bm25_b)
        if score > 0:
            scored.append((score, doc_idx, record))

    scored.sort(key=lambda item: (-item[0], item[1]))
    limit = max(top_k, 0)
    hits = [(score, record) for score, _, record in scored[:limit]]
    logger.info(
        "rag.lexical.query rag=%s top_k=%s returned=%s",
        rag_slug,
        top_k,
        len(hits),
        extra={
            "rag_slug": rag_slug,
            "requested_top_k": top_k,
            "returned": len(hits),
        },
    )
    return hits

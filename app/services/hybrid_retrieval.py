# generated-by: codex-agent 2026-05-05T00:00:00Z
"""
Hybrid RAG retrieval using semantic FAISS hits plus local BM25 hits.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Callable, Sequence

from app.models.admin import RetrievalConfig
from app.services.runtime_config import get_retrieval_config_sync
from app.services.lexical_search import search_lexical_index
from app.services.vectorstore import ChunkRecord, query_index

logger = logging.getLogger("chatfleet.hybrid_retrieval")
logger.setLevel(logging.INFO)

RRF_K = 60
SEMANTIC_WEIGHT = 1.0
LEXICAL_WEIGHT = 1.0

ScoredHit = tuple[float, ChunkRecord]
ChunkKey = tuple[str, int]
SemanticSearch = Callable[[str, Sequence[float], int, float | None], list[ScoredHit]]
LexicalSearch = Callable[..., list[ScoredHit]]


@dataclass(frozen=True)
class HybridDiagnostics:
    semantic_count: int
    lexical_count: int
    final_count: int
    semantic_ranks: dict[ChunkKey, int]
    lexical_ranks: dict[ChunkKey, int]
    index_missing: bool = False
    index_error: str | None = None
    candidate_k: int = 0
    semantic_ms: float = 0.0
    lexical_ms: float = 0.0
    fusion_ms: float = 0.0
    retrieval_ms: float = 0.0
    embedding_ms: float = 0.0


@dataclass(frozen=True)
class HybridRetrievalResult:
    hits: list[ScoredHit]
    diagnostics: HybridDiagnostics


def chunk_key(record: ChunkRecord) -> ChunkKey:
    return (record.doc_id, record.chunk_index)


def _rank_map(hits: Sequence[ScoredHit]) -> dict[ChunkKey, int]:
    ranks: dict[ChunkKey, int] = {}
    for rank, (_, record) in enumerate(hits, start=1):
        ranks.setdefault(chunk_key(record), rank)
    return ranks


def reciprocal_rank_fusion(
    semantic_hits: Sequence[ScoredHit],
    lexical_hits: Sequence[ScoredHit],
    top_k: int,
    *,
    rrf_k: int = RRF_K,
    semantic_weight: float = SEMANTIC_WEIGHT,
    lexical_weight: float = LEXICAL_WEIGHT,
) -> list[ScoredHit]:
    """Fuse ranked hit lists with deterministic de-duplication."""

    combined: dict[ChunkKey, dict[str, object]] = {}
    first_seen = 0

    def add_hits(hits: Sequence[ScoredHit], weight: float) -> None:
        nonlocal first_seen
        if weight <= 0:
            return
        for rank, (_, record) in enumerate(hits, start=1):
            key = chunk_key(record)
            if key not in combined:
                combined[key] = {
                    "score": 0.0,
                    "record": record,
                    "first_seen": first_seen,
                }
                first_seen += 1
            combined[key]["score"] = float(combined[key]["score"]) + (
                weight / (rrf_k + rank)
            )

    add_hits(semantic_hits, semantic_weight)
    add_hits(lexical_hits, lexical_weight)

    ranked = sorted(
        combined.values(),
        key=lambda item: (
            -float(item["score"]),
            int(item["first_seen"]),
            chunk_key(item["record"]),  # type: ignore[arg-type]
        ),
    )
    limit = max(top_k, 0)
    return [
        (float(item["score"]), item["record"])  # type: ignore[arg-type]
        for item in ranked[:limit]
    ]


def hybrid_retrieve(
    rag_slug: str,
    query_vector: Sequence[float],
    question: str,
    top_k: int,
    *,
    min_semantic_score: float | None,
    semantic_search: SemanticSearch = query_index,
    lexical_search: LexicalSearch = search_lexical_index,
    retrieval_config: RetrievalConfig | None = None,
) -> HybridRetrievalResult:
    """
    Retrieve semantic and lexical candidates, then fuse them via RRF.

    If the FAISS index is absent, no context is available. If BM25 fails, the
    semantic candidate list is returned unchanged as the compatibility fallback.
    """

    started = time.perf_counter()
    cfg = retrieval_config or get_retrieval_config_sync()
    effective_min_score = (
        cfg.semantic_min_score if min_semantic_score is None else min_semantic_score
    )
    candidate_k = max(top_k * cfg.candidate_multiplier, cfg.candidate_min)
    semantic_started = time.perf_counter()
    try:
        semantic_hits = semantic_search(
            rag_slug,
            query_vector,
            candidate_k,
            effective_min_score,
        )
    except FileNotFoundError as exc:
        semantic_ms = (time.perf_counter() - semantic_started) * 1000.0
        logger.warning(
            "rag.hybrid.index_missing rag=%s error=%s",
            rag_slug,
            exc,
            extra={"rag_slug": rag_slug, "index_error": str(exc)},
        )
        return HybridRetrievalResult(
            hits=[],
            diagnostics=HybridDiagnostics(
                semantic_count=0,
                lexical_count=0,
                final_count=0,
                semantic_ranks={},
                lexical_ranks={},
                index_missing=True,
                index_error=str(exc),
                candidate_k=candidate_k,
                semantic_ms=round(semantic_ms, 3),
                retrieval_ms=round((time.perf_counter() - started) * 1000.0, 3),
            ),
        )
    semantic_ms = (time.perf_counter() - semantic_started) * 1000.0

    if cfg.mode == "semantic":
        final_hits = semantic_hits[: max(top_k, 0)]
        return HybridRetrievalResult(
            hits=final_hits,
            diagnostics=HybridDiagnostics(
                semantic_count=len(semantic_hits),
                lexical_count=0,
                final_count=len(final_hits),
                semantic_ranks=_rank_map(semantic_hits),
                lexical_ranks={},
                candidate_k=candidate_k,
                semantic_ms=round(semantic_ms, 3),
                retrieval_ms=round((time.perf_counter() - started) * 1000.0, 3),
            ),
        )

    lexical_started = time.perf_counter()
    try:
        lexical_hits = lexical_search(
            rag_slug,
            question,
            candidate_k,
            bm25_k1=cfg.bm25_k1,
            bm25_b=cfg.bm25_b,
        )
    except FileNotFoundError:
        lexical_hits = []
    except Exception:
        lexical_ms = (time.perf_counter() - lexical_started) * 1000.0
        logger.exception(
            "rag.hybrid.lexical_failed rag=%s fallback=semantic_only",
            rag_slug,
            extra={"rag_slug": rag_slug, "retrieval_fallback": "semantic_only"},
        )
        final_hits = semantic_hits[: max(top_k, 0)]
        return HybridRetrievalResult(
            hits=final_hits,
            diagnostics=HybridDiagnostics(
                semantic_count=len(semantic_hits),
                lexical_count=0,
                final_count=len(final_hits),
                semantic_ranks=_rank_map(semantic_hits),
                lexical_ranks={},
                candidate_k=candidate_k,
                semantic_ms=round(semantic_ms, 3),
                lexical_ms=round(lexical_ms, 3),
                retrieval_ms=round((time.perf_counter() - started) * 1000.0, 3),
            ),
        )
    lexical_ms = (time.perf_counter() - lexical_started) * 1000.0

    fusion_started = time.perf_counter()
    final_hits = reciprocal_rank_fusion(
        semantic_hits,
        lexical_hits,
        top_k,
        rrf_k=cfg.rrf_k,
        semantic_weight=cfg.semantic_weight,
        lexical_weight=cfg.lexical_weight,
    )
    fusion_ms = (time.perf_counter() - fusion_started) * 1000.0
    return HybridRetrievalResult(
        hits=final_hits,
        diagnostics=HybridDiagnostics(
            semantic_count=len(semantic_hits),
            lexical_count=len(lexical_hits),
            final_count=len(final_hits),
            semantic_ranks=_rank_map(semantic_hits),
            lexical_ranks=_rank_map(lexical_hits),
            candidate_k=candidate_k,
            semantic_ms=round(semantic_ms, 3),
            lexical_ms=round(lexical_ms, 3),
            fusion_ms=round(fusion_ms, 3),
            retrieval_ms=round((time.perf_counter() - started) * 1000.0, 3),
        ),
    )

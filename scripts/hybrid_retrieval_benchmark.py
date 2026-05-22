# generated-by: codex-agent 2026-05-05T00:00:00Z
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import tempfile
import time
from pathlib import Path
from statistics import median
from typing import Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.services import lexical_search, vectorstore  # noqa: E402
from app.services.hybrid_retrieval import hybrid_retrieve  # noqa: E402
from app.services.lexical_search import search_lexical_index  # noqa: E402
from app.services.vectorstore import (  # noqa: E402
    build_index,
    persist_doc_payload,
    query_index,
)

RAG_SLUG = "hybrid-benchmark"
DOC_ID = "22222222-2222-2222-2222-222222222222"
TARGET_REF = "R-2024-17"
DIM = 32
QUERY_VECTOR = [1.0] + [0.0] * (DIM - 1)


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, math.ceil((pct / 100.0) * len(ordered)) - 1)
    return ordered[idx]


def _measure_ms(fn: Callable[[], object], iterations: int) -> list[float]:
    timings: list[float] = []
    for _ in range(iterations):
        start = time.perf_counter()
        fn()
        timings.append((time.perf_counter() - start) * 1000.0)
    return timings


def _summarize_ms(values: list[float]) -> dict[str, float]:
    return {
        "min_ms": round(min(values), 3) if values else 0.0,
        "p50_ms": round(median(values), 3) if values else 0.0,
        "p95_ms": round(_percentile(values, 95), 3) if values else 0.0,
        "max_ms": round(max(values), 3) if values else 0.0,
    }


def _embedding_for(index: int) -> list[float]:
    vec = [0.0] * DIM
    if index == 17:
        # Keep the exact-reference chunk away from semantic top-1 so the
        # benchmark proves the lexical leg matters.
        vec[0] = 0.95
        vec[1] = 0.31
    elif index % 5000 == 0:
        vec[0] = 0.96
        vec[2] = 0.28
    elif index % 7 == 0:
        vec[0] = 0.72
        vec[3] = 0.69
    else:
        axis = 4 + (index % (DIM - 4))
        vec[axis] = 1.0
        vec[0] = 0.05 + ((index % 5) * 0.02)
    return vec


def _chunk_text(index: int) -> str:
    if index == 17:
        return (
            "Procédure disciplinaire R-2024-17: la convocation doit mentionner "
            "le motif précis et laisser cinq jours ouvrables avant l'entretien."
        )
    if index % 5000 == 0:
        return (
            f"Congé parental dossier RH-{index:06d}: les salariés éligibles "
            "peuvent demander seize semaines avec préavis écrit."
        )
    if index % 7 == 0:
        return (
            f"Télétravail accord TW-{index:06d}: deux jours par semaine sont "
            "autorisés après validation du responsable."
        )
    if index % 5 == 0:
        return (
            f"Paie 2026 code PAY-{index:06d}: les primes exceptionnelles sont "
            "versées au cycle mensuel validé."
        )
    return (
        f"Note juridique GEN-{index:06d}: clause standard, obligations de suivi, "
        "contrôle interne et archivage documentaire."
    )


def _build_corpus(size: int) -> tuple[list[dict[str, object]], list[list[float]]]:
    chunks = [
        {
            "text": _chunk_text(idx),
            "page_start": (idx % 250) + 1,
            "page_end": (idx % 250) + 1,
        }
        for idx in range(size)
    ]
    embeddings = [_embedding_for(idx) for idx in range(size)]
    return chunks, embeddings


def _benchmark_size(size: int, queries: int, top_k: int, keep_dir: Path | None) -> dict:
    temp_dir = Path(tempfile.mkdtemp(prefix=f"chatfleet-hybrid-bench-{size}-"))
    index_dir = temp_dir / "indexes"
    upload_dir = temp_dir / "uploads"
    index_dir.mkdir(parents=True, exist_ok=True)
    upload_dir.mkdir(parents=True, exist_ok=True)

    original_runtime = vectorstore.get_runtime_overrides_sync
    vectorstore.get_runtime_overrides_sync = lambda: (
        index_dir,
        upload_dir,
        50,
        0.2,
        top_k,
    )
    lexical_search._CACHE.clear()

    try:
        chunks, embeddings = _build_corpus(size)
        build_start = time.perf_counter()
        persist_doc_payload(
            RAG_SLUG,
            DOC_ID,
            f"synthetic-{size}.pdf",
            chunks,
            embeddings,
        )
        total_chunks, dimension = build_index(RAG_SLUG)
        build_seconds = time.perf_counter() - build_start

        semantic_exact = query_index(
            RAG_SLUG,
            QUERY_VECTOR,
            top_k=1,
            min_score=0.0,
        )
        lexical_search._CACHE.clear()
        cold_lexical_start = time.perf_counter()
        lexical_cold = search_lexical_index(
            RAG_SLUG,
            f"Que prévoit la référence {TARGET_REF} ?",
            top_k,
        )
        lexical_cold_ms = (time.perf_counter() - cold_lexical_start) * 1000.0

        lexical_warm_ms = _measure_ms(
            lambda: search_lexical_index(
                RAG_SLUG,
                f"Que prévoit la référence {TARGET_REF} ?",
                top_k,
            ),
            queries,
        )
        semantic_ms = _measure_ms(
            lambda: query_index(RAG_SLUG, QUERY_VECTOR, top_k, min_score=0.0),
            queries,
        )

        lexical_search._CACHE.clear()
        cold_hybrid_start = time.perf_counter()
        hybrid_cold = hybrid_retrieve(
            RAG_SLUG,
            QUERY_VECTOR,
            f"Que prévoit la référence {TARGET_REF} ?",
            top_k,
            min_semantic_score=0.0,
        )
        hybrid_cold_ms = (time.perf_counter() - cold_hybrid_start) * 1000.0

        hybrid_warm_ms = _measure_ms(
            lambda: hybrid_retrieve(
                RAG_SLUG,
                QUERY_VECTOR,
                f"Que prévoit la référence {TARGET_REF} ?",
                top_k,
                min_semantic_score=0.0,
            ),
            queries,
        )

        return {
            "size": size,
            "queries": queries,
            "top_k": top_k,
            "chunks": total_chunks,
            "dimension": dimension,
            "build_seconds": round(build_seconds, 3),
            "semantic_top1_contains_ref": TARGET_REF in semantic_exact[0][1].text,
            "lexical_top1_contains_ref": bool(lexical_cold)
            and TARGET_REF in lexical_cold[0][1].text,
            "hybrid_top1_contains_ref": bool(hybrid_cold.hits)
            and TARGET_REF in hybrid_cold.hits[0][1].text,
            "lexical_cold_ms": round(lexical_cold_ms, 3),
            "hybrid_cold_ms": round(hybrid_cold_ms, 3),
            "semantic": _summarize_ms(semantic_ms),
            "lexical_warm": _summarize_ms(lexical_warm_ms),
            "hybrid_warm": _summarize_ms(hybrid_warm_ms),
            "work_dir": str(temp_dir) if keep_dir else None,
        }
    finally:
        vectorstore.get_runtime_overrides_sync = original_runtime
        lexical_search._CACHE.clear()
        if keep_dir:
            keep_dir.mkdir(parents=True, exist_ok=True)
            target = keep_dir / temp_dir.name
            if target.exists():
                shutil.rmtree(target)
            shutil.move(str(temp_dir), target)
        else:
            shutil.rmtree(temp_dir, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark local hybrid RAG retrieval on synthetic corpora."
    )
    parser.add_argument(
        "--sizes",
        nargs="+",
        type=int,
        default=[1000, 10000, 50000],
        help="Corpus sizes in chunks.",
    )
    parser.add_argument("--queries", type=int, default=30)
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--keep-dir", type=Path, default=None)
    args = parser.parse_args()

    results = [
        _benchmark_size(size, args.queries, args.top_k, args.keep_dir)
        for size in args.sizes
    ]
    print(json.dumps({"results": results}, indent=2))


if __name__ == "__main__":
    main()

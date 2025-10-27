# generated-by: codex-agent 2025-02-15T00:45:00Z
"""
Embedding helpers with deterministic fallback for offline testing.
"""

from __future__ import annotations

import asyncio
import hashlib
import math
import os
from typing import Iterable, List

import numpy as np

from app.core.config import settings

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - optional dependency
    OpenAI = None  # type: ignore

_openai_client: OpenAI | None = None
EMBED_DIM = 1536


def _ensure_client() -> OpenAI | None:
    global _openai_client
    if _openai_client is not None:
        return _openai_client
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or OpenAI is None:
        return None
    _openai_client = OpenAI(api_key=api_key)
    return _openai_client


def _deterministic_embedding(text: str, dim: int = EMBED_DIM) -> List[float]:
    """Generate a deterministic pseudo-embedding using SHA256."""

    digest = hashlib.sha256(text.encode("utf-8")).digest()
    arr = np.frombuffer(digest, dtype=np.uint8).astype(np.float32)
    repeats = math.ceil(dim / arr.size)
    tiled = np.tile(arr, repeats)[:dim]
    norm = np.linalg.norm(tiled)
    if norm == 0:
        return tiled.tolist()
    return (tiled / norm).tolist()


async def embed_texts(texts: Iterable[str]) -> List[List[float]]:
    """Return embeddings for the given texts, using OpenAI when available."""

    items = list(texts)
    if not items:
        return []

    client = _ensure_client()
    if client is None:
        return [_deterministic_embedding(text) for text in items]

    loop = asyncio.get_running_loop()

    def _call() -> List[List[float]]:
        response = client.embeddings.create(model=settings.embed_model, input=list(items))
        return [row.embedding for row in response.data]

    try:
        return await loop.run_in_executor(None, _call)
    except Exception:
        return [_deterministic_embedding(text) for text in items]


async def embed_text(text: str) -> List[float]:
    (vector,) = await embed_texts([text])
    return vector

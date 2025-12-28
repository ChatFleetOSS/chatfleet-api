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
from app.services.runtime_config import get_llm_config, get_api_key

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

    # prefer runtime configuration when available
    client = _ensure_client()
    if client is None:
        try:
            cfg = await get_llm_config()
            key = os.getenv("OPENAI_API_KEY") or (await get_api_key())
            # For vLLM, allow missing key by providing a dummy value.
            eff_key = key or ("sk-ignored" if cfg.provider == "vllm" else None)
            if eff_key and OpenAI is not None:
                client = OpenAI(api_key=eff_key, base_url=cfg.base_url)  # type: ignore[call-arg]
        except Exception:
            client = None
    if client is None:
        return [_deterministic_embedding(text) for text in items]

    loop = asyncio.get_running_loop()

    def _call(model_name: str) -> List[List[float]]:
        response = client.embeddings.create(model=model_name, input=list(items))
        return [row.embedding for row in response.data]

    try:
        cfg = await get_llm_config()
        model_name = cfg.embed_model or settings.embed_model
        return await loop.run_in_executor(None, lambda: _call(model_name))
    except Exception:
        return [_deterministic_embedding(text) for text in items]


async def embed_text(text: str) -> List[float]:
    (vector,) = await embed_texts([text])
    return vector


async def test_embedding_provider(payload: "LLMConfigTestRequest") -> tuple[bool, str | None, int | None]:
    """Verify the embeddings operation for the configured provider.

    Returns (ok, message, dim).
    """
    try:
        # Lazy import to avoid circular import at module import time
        from app.models.admin import LLMConfigTestRequest  # noqa: F401
    except Exception:
        pass

    if OpenAI is None:
        return False, "Client not available: install openai package", None

    provider = payload.provider
    base_url = payload.base_url
    key = payload.api_key or os.getenv("OPENAI_API_KEY") or (await get_api_key())

    if provider == "openai" and not key:
        return False, "AUTH_ERROR: missing API key", None
    if provider == "vllm" and not base_url:
        return False, "CONFIG_ERROR: missing base_url for vLLM", None

    eff_key = key or ("sk-ignored" if provider == "vllm" else "")
    try:
        client = OpenAI(api_key=eff_key, base_url=base_url)  # type: ignore[call-arg]
        loop = asyncio.get_running_loop()

        def _call() -> int:
            resp = client.embeddings.create(
                model=(payload.embed_model or settings.embed_model),
                input=["hello world"],
            )
            return len(resp.data[0].embedding)

        dim = await asyncio.wait_for(loop.run_in_executor(None, _call), timeout=15)
        return True, None, int(dim)
    except Exception as exc:  # pragma: no cover
        msg = str(exc)
        if "401" in msg or "Unauthorized" in msg:
            return False, "AUTH_ERROR: unauthorized", None
        if "timed out" in msg or "Timeout" in msg:
            return False, "TIMEOUT: provider did not respond", None
        if "Not Found" in msg or "404" in msg:
            return False, "INVALID_ENDPOINT: check base_url", None
        if "model" in msg and "not found" in msg.lower():
            return False, "INVALID_MODEL: check embedding model id", None
        return False, f"PROVIDER_ERROR: {msg}", None


async def prewarm_embeddings() -> None:
    """Trigger a small embeddings call to warm caches/startup."""
    try:
        _ = await embed_text("hello")
    except Exception:
        # Non-fatal; warmup is best effort
        pass

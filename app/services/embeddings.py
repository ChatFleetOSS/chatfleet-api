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
import logging

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from app.core.config import settings
from app.services.runtime_config import get_llm_config, get_api_key

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - optional dependency
    OpenAI = None  # type: ignore

try:
    from sentence_transformers import SentenceTransformer
except ImportError:  # pragma: no cover - optional dependency
    SentenceTransformer = None  # type: ignore

_client_cache: dict[tuple[str | None, str | None], OpenAI] = {}
_local_models: dict[str, "SentenceTransformer"] = {}
EMBED_DIM = 1536
LOCAL_EMBED_MODEL_DEFAULT = "BAAI/bge-m3"
LOCAL_EMBED_MODEL_DEFAULT_DIM = 1024
logger = logging.getLogger("chatfleet.embeddings")
logger.setLevel(logging.INFO)


def _get_embed_client(provider: str, base_url: str | None, key: str | None) -> OpenAI | None:
    if OpenAI is None:
        return None
    eff_key = key or ("sk-ignored" if provider == "vllm" else None)
    if not eff_key:
        return None
    cache_key = (base_url, eff_key)
    cached = _client_cache.get(cache_key)
    if cached is not None:
        return cached
    client = OpenAI(api_key=eff_key, base_url=base_url)  # type: ignore[call-arg]
    _client_cache[cache_key] = client
    return client


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

def _ensure_local_model(model_name: str) -> "SentenceTransformer" | None:
    if SentenceTransformer is None:
        return None
    model = _local_models.get(model_name)
    if model is not None:
        return model
    model = SentenceTransformer(model_name)
    _local_models[model_name] = model
    return model

async def _embed_texts_local(texts: list[str], model_name: str) -> List[List[float]]:
    model = _ensure_local_model(model_name)
    if model is None:
        raise RuntimeError("Client not available: install sentence-transformers")
    loop = asyncio.get_running_loop()

    def _call() -> List[List[float]]:
        vecs = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
        vecs = vecs.astype("float32")
        return vecs.tolist()

    return await loop.run_in_executor(None, _call)


def _fallback_dim(cfg) -> int:
    """Best-effort guess for embedding dimension to avoid mixed dims on fallback."""
    try:
        if cfg and getattr(cfg, "embed_provider", "openai") == "local":
            model_name = getattr(cfg, "embed_model", None) or LOCAL_EMBED_MODEL_DEFAULT
            model = _local_models.get(model_name)
            if model and hasattr(model, "get_sentence_embedding_dimension"):
                return int(model.get_sentence_embedding_dimension())  # type: ignore[call-arg]
            if model_name == LOCAL_EMBED_MODEL_DEFAULT:
                return LOCAL_EMBED_MODEL_DEFAULT_DIM
    except Exception:
        pass
    return EMBED_DIM


async def embed_texts(texts: Iterable[str]) -> List[List[float]]:
    """Return embeddings for the given texts, using OpenAI when available."""

    items = list(texts)
    if not items:
        return []

    # prefer runtime configuration when available
    cfg = None
    try:
        cfg = await get_llm_config()
        if getattr(cfg, "embed_provider", "openai") == "local":
            model_name = cfg.embed_model or LOCAL_EMBED_MODEL_DEFAULT
            logger.info("embeddings.local", extra={"count": len(items), "model": model_name})
            vectors = await _embed_texts_local(items, model_name)
            logger.info(
                "embeddings.local.ok",
                extra={"count": len(vectors), "dim": len(vectors[0]) if vectors else 0},
            )
            return vectors
    except Exception:
        cfg = None

    key = os.getenv("OPENAI_API_KEY")
    if cfg is not None:
        try:
            key = key or (await get_api_key())
        except Exception:
            pass
    provider = getattr(cfg, "provider", "openai")
    base_url = None if provider == "openai" else getattr(cfg, "base_url", None)
    client = _get_embed_client(provider, base_url, key)
    if client is None:
        dim = _fallback_dim(cfg)
        logger.warning("embeddings.fallback.deterministic", extra={"count": len(items), "dim": dim})
        vectors = [_deterministic_embedding(text, dim=dim) for text in items]
        logger.info(
            "embeddings.fallback.ok",
            extra={"count": len(vectors), "dim": len(vectors[0]) if vectors else 0},
        )
        return vectors

    loop = asyncio.get_running_loop()

    def _call(model_name: str) -> List[List[float]]:
        response = client.embeddings.create(model=model_name, input=list(items))
        return [row.embedding for row in response.data]

    try:
        cfg = cfg or await get_llm_config()
        if getattr(cfg, "embed_provider", "openai") == "local":
            model_name = cfg.embed_model or LOCAL_EMBED_MODEL_DEFAULT
            logger.info("embeddings.local", extra={"count": len(items), "model": model_name})
            return await _embed_texts_local(items, model_name)
        model_name = cfg.embed_model or settings.embed_model
        logger.info("embeddings.remote", extra={"count": len(items), "model": model_name, "provider": cfg.provider})
        vectors = await loop.run_in_executor(None, lambda: _call(model_name))
        logger.info(
            "embeddings.remote.ok",
            extra={"count": len(vectors), "dim": len(vectors[0]) if vectors else 0, "provider": cfg.provider},
        )
        return vectors
    except Exception:
        logger.exception("embeddings.error", extra={"count": len(items)})
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

    embed_provider = payload.embed_provider or ("local" if payload.provider == "vllm" else "openai")
    if embed_provider == "local":
        model_name = payload.embed_model or LOCAL_EMBED_MODEL_DEFAULT
        try:
            vectors = await _embed_texts_local(["hello world"], model_name)
            dim = len(vectors[0]) if vectors else None
            return True, None, dim
        except Exception as exc:
            return False, f"PROVIDER_ERROR: {exc}", None

    if OpenAI is None:
        return False, "Client not available: install openai package", None

    provider = payload.provider
    key = payload.api_key or os.getenv("OPENAI_API_KEY") or (await get_api_key())
    base_url = payload.base_url if provider == "openai" else None

    if not key:
        return False, "AUTH_ERROR: missing API key", None

    try:
        client = OpenAI(api_key=key, base_url=base_url)  # type: ignore[call-arg]
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

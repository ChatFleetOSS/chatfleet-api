"""
Runtime-configurable LLM and embeddings settings with optional secret encryption.
"""

from __future__ import annotations

import base64
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from cryptography.fernet import Fernet, InvalidToken
from motor.motor_asyncio import AsyncIOMotorCollection

from app.core.database import get_collection
from app.models.admin import LLMConfigUpdateRequest, LLMConfigView
from app.core.config import settings


_CACHE: Optional[Tuple[LLMConfigView, datetime]] = None


def _col() -> AsyncIOMotorCollection:
    return get_collection("admin_settings")


def _fernet() -> Optional[Fernet]:
    key = os.getenv("CONFIG_MASTER_KEY")
    if not key:
        return None
    # Accept raw 32-byte base64 key or arbitrary string we derive to 32 bytes
    try:
        # If key looks like urlsafe base64 of 32 bytes, use it as-is
        Fernet(key)  # type: ignore[arg-type]
        return Fernet(key)  # type: ignore[arg-type]
    except Exception:
        # Derive a deterministic 32-byte key from provided string
        padded = base64.urlsafe_b64encode(key.encode("utf-8")).ljust(44, b"=")
        return Fernet(padded)


def _enc(secret: Optional[str]) -> Optional[str]:
    if not secret:
        return None
    f = _fernet()
    if not f:
        # dev-only fallback: store as plain base64 so it's at least not human-readable
        return base64.urlsafe_b64encode(secret.encode("utf-8")).decode("utf-8")
    token = f.encrypt(secret.encode("utf-8"))
    return token.decode("utf-8")


def _dec(token: Optional[str]) -> Optional[str]:
    if not token:
        return None
    f = _fernet()
    if not f:
        try:
            return base64.urlsafe_b64decode(token.encode("utf-8")).decode("utf-8")
        except Exception:
            return None
    try:
        return f.decrypt(token.encode("utf-8")).decode("utf-8")
    except InvalidToken:
        return None


async def ensure_indexes() -> None:
    col = _col()
    # MongoDB enforces a unique index on _id automatically; creating one with
    # options raises an error. Keep a lightweight index on verified_at to
    # support potential future queries, but it's not required.
    try:
        await col.create_index("verified_at")
    except Exception:
        # Defensive: ignore index creation failures in dev environments.
        pass


async def get_llm_config() -> LLMConfigView:
    global _CACHE
    if _CACHE is not None:
        return _CACHE[0]
    col = _col()
    doc = await col.find_one({"_id": "runtime"}) or {}
    provider = doc.get("provider", "openai")
    default_embed_provider = "local" if provider == "vllm" else "openai"
    embed_provider = doc.get("embed_provider", default_embed_provider)
    if provider == "vllm":
        embed_provider = "local"

    cfg = LLMConfigView(
        provider=provider,
        base_url=doc.get("base_url"),
        chat_model=doc.get("chat_model", "gpt-4o-mini"),
        embed_model=doc.get("embed_model", "BAAI/bge-m3"),
        embed_provider=embed_provider,
        temperature_default=float(doc.get("temperature_default", 0.2)),
        top_k_default=int(doc.get("top_k_default", 12)),
        index_dir=str(doc.get("index_dir", settings.index_dir)),
        upload_dir=str(doc.get("upload_dir", settings.upload_dir)),
        max_upload_mb=int(doc.get("max_upload_mb", settings.max_upload_mb)),
        has_api_key=bool(doc.get("api_key_enc")),
        verified_at=doc.get("verified_at"),
        runtime_enabled=True,
    )
    _CACHE = (cfg, datetime.now(timezone.utc))
    return cfg


def _invalidate_cache() -> None:
    global _CACHE
    _CACHE = None


async def set_llm_config(payload: LLMConfigUpdateRequest, actor_id: str) -> LLMConfigView:
    # Enforce local embeddings when using vLLM to avoid OpenAI dependency mismatches.
    embed_provider = payload.embed_provider
    if payload.provider == "vllm":
        embed_provider = "local"
    col = _col()
    doc: Dict[str, Any] = {
        "_id": "runtime",
        "provider": payload.provider,
        "base_url": payload.base_url,
        "chat_model": payload.chat_model,
        "embed_model": payload.embed_model,
        "embed_provider": embed_provider,
        "embed_provider": payload.embed_provider,
        "temperature_default": payload.temperature_default if payload.temperature_default is not None else 0.2,
        "top_k_default": payload.top_k_default if payload.top_k_default is not None else 12,
        "index_dir": str(payload.index_dir) if payload.index_dir is not None else str(settings.index_dir),
        "upload_dir": str(payload.upload_dir) if payload.upload_dir is not None else str(settings.upload_dir),
        "max_upload_mb": int(payload.max_upload_mb) if payload.max_upload_mb is not None else int(settings.max_upload_mb),
        "updated_by": actor_id,
        "updated_at": datetime.now(timezone.utc),
    }
    update: Dict[str, Any] = {"$set": doc}
    if payload.api_key is not None:
        if payload.api_key == "":
            update["$unset"] = {"api_key_enc": ""}
        else:
            update["$set"]["api_key_enc"] = _enc(payload.api_key)
    await col.update_one({"_id": "runtime"}, update, upsert=True)
    _invalidate_cache()
    cfg = await get_llm_config()
    return cfg


async def get_api_key() -> Optional[str]:
    col = _col()
    doc = await col.find_one({"_id": "runtime"}) or {}
    return _dec(doc.get("api_key_enc"))


async def set_verified(ok: bool) -> None:
    if not ok:
        return
    col = _col()
    await col.update_one({"_id": "runtime"}, {"$set": {"verified_at": datetime.now(timezone.utc)}})
    _invalidate_cache()


def get_runtime_overrides_sync() -> tuple[Path, Path, int, float, int]:
    """Return (index_dir, upload_dir, max_upload_mb, temperature_default, top_k_default).

    Synchronous accessor used by modules that cannot `await`. Falls back to
    environment-backed settings if runtime cache is not warmed.
    """
    cfg = _CACHE[0] if _CACHE else None
    index_dir = Path(cfg.index_dir) if cfg and getattr(cfg, "index_dir", None) else settings.index_dir
    upload_dir = Path(cfg.upload_dir) if cfg and getattr(cfg, "upload_dir", None) else settings.upload_dir
    max_upload_mb = int(cfg.max_upload_mb) if cfg and getattr(cfg, "max_upload_mb", None) else int(settings.max_upload_mb)
    temperature_default = float(cfg.temperature_default) if cfg and getattr(cfg, "temperature_default", None) else float(settings.temperature_default)
    top_k_default = int(cfg.top_k_default) if cfg and getattr(cfg, "top_k_default", None) else int(settings.top_k_default)
    return index_dir, upload_dir, max_upload_mb, temperature_default, top_k_default

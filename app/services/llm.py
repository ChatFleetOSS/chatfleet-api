# generated-by: codex-agent 2025-02-17T12:40:00Z
"""
Large language model helpers for answer generation.

The implementation prefers OpenAI's Chat Completions API when credentials are
available, and falls back to `None` to let callers apply deterministic logic.
"""

from __future__ import annotations

import asyncio
import os
from typing import Dict, List, Optional, Tuple
from functools import partial

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - optional dependency
    OpenAI = None  # type: ignore

from app.core.config import settings
from app.models.admin import LLMConfigTestRequest
from app.services.runtime_config import get_llm_config, get_api_key

_client: OpenAI | None = None


def _ensure_client_env_only() -> OpenAI | None:
    global _client
    if _client is not None:
        return _client
    # prefer runtime config key
    api_key = os.getenv("OPENAI_API_KEY")
    base_url: str | None = None
    # do not attempt async operations here; environment-only client
    if not api_key or OpenAI is None:
        return None
    _client = OpenAI(api_key=api_key)
    return _client


async def generate_chat_completion(
    messages: List[Dict[str, str]],
    *,
    temperature: float,
    max_tokens: int,
) -> Optional[Tuple[str, int]]:
    """
    Call the configured chat model to obtain an answer.

    Returns (answer_text, completion_tokens) when successful, or None when the
    client is unavailable or the API call fails.
    """

    # prefer runtime configuration when available
    try:
        from app.services.runtime_config import get_llm_config, get_api_key as _get_key
        cfg = await get_llm_config()
        key = os.getenv("OPENAI_API_KEY") or (await _get_key())
        # For vLLM, an API key may be optional but OpenAI client requires a non-empty string.
        eff_key = key or ("sk-ignored" if cfg.provider == "vllm" else None)
        if eff_key and OpenAI is not None:
            client = OpenAI(api_key=eff_key, base_url=cfg.base_url)  # type: ignore[call-arg]
        else:
            client = _ensure_client_env_only()
    except Exception:
        client = _ensure_client_env_only()
    if client is None:
        return None

    loop = asyncio.get_running_loop()

    def _call(model_name: str) -> Tuple[str, int]:
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if not response.choices:
            return "", 0
        choice = response.choices[0]
        text = (choice.message.content or "").strip()
        completion_tokens = 0
        if getattr(response, "usage", None) is not None:
            completion_tokens = getattr(response.usage, "completion_tokens", 0) or 0
        if completion_tokens <= 0:
            # Fallback estimate when the API does not return usage statistics.
            completion_tokens = max(1, len(text.split()))
        return text, completion_tokens

    try:
        cfg = await get_llm_config()
        model_name = cfg.chat_model or settings.chat_model
        return await loop.run_in_executor(None, partial(_call, model_name))
    except Exception:
        return None


async def test_chat_completion_provider(payload: LLMConfigTestRequest) -> tuple[bool, str | None]:
    """Verify connectivity and basic capability to the configured provider.

    - For provider 'openai': require an API key; attempt `models.list()` first, then a 1-token completion.
    - For provider 'vllm': require a base_url; use a dummy key if none provided; attempt `models.list()`.
    A strict overall timeout of ~15s is enforced by running calls in a thread and awaiting with wait_for.
    """
    if OpenAI is None:
        return False, "Client not available: install openai package"

    provider = payload.provider
    base_url = payload.base_url
    key = payload.api_key or os.getenv("OPENAI_API_KEY") or (await get_api_key())

    if provider == "openai" and not key:
        return False, "AUTH_ERROR: missing API key"
    if provider == "vllm" and not base_url:
        return False, "CONFIG_ERROR: missing base_url for vLLM"

    eff_key = key or ("sk-ignored" if provider == "vllm" else "")
    try:
        client = OpenAI(api_key=eff_key, base_url=base_url)  # type: ignore[call-arg]

        loop = asyncio.get_running_loop()

        def _list_models() -> bool:
            _ = client.models.list()
            return True

        async def _run_with_timeout(fn) -> bool:
            return await asyncio.wait_for(loop.run_in_executor(None, fn), timeout=15)

        try:
            ok = await _run_with_timeout(_list_models)
            if ok:
                return True, None
        except Exception:
            # Try a tiny completion for providers that support chat
            try:
                def _tiny_completion() -> bool:
                    _ = client.chat.completions.create(
                        model=(payload.chat_model or settings.chat_model),
                        messages=[{"role": "user", "content": "ping"}],
                        max_tokens=1,
                        temperature=0,
                    )
                    return True

                ok2 = await _run_with_timeout(_tiny_completion)
                if ok2:
                    return True, None
            except Exception as exc2:
                # Map common errors
                msg = str(exc2)
                if "401" in msg or "Unauthorized" in msg:
                    return False, "AUTH_ERROR: unauthorized"
                if "timed out" in msg or "Timeout" in msg:
                    return False, "TIMEOUT: provider did not respond"
                if "Not Found" in msg or "404" in msg:
                    return False, "INVALID_ENDPOINT: check base_url"
                return False, f"PROVIDER_ERROR: {msg}"

        # If we got here, listing failed without exception context
        return False, "CONNECTION_ERROR: unable to reach provider"
    except Exception as exc:  # pragma: no cover
        msg = str(exc)
        if "API key" in msg and (provider == "openai"):
            return False, "AUTH_ERROR: missing or invalid API key"
        if "Name or service not known" in msg or "Failed to establish a new connection" in msg:
            return False, "CONNECTION_ERROR: check base_url/network"
        return False, f"PROVIDER_ERROR: {msg}"


async def discover_models(provider: str, base_url: Optional[str], api_key: Optional[str]) -> tuple[list[str], list[str], list[str]]:
    """Return (chat_models, embedding_models, raw_models).

    Uses the OpenAI-compatible /v1/models when available. For vLLM, base_url is required.
    """
    if OpenAI is None:
        return [], [], []
    if provider == "vllm" and not base_url:
        return [], [], []

    key = api_key or os.getenv("OPENAI_API_KEY") or (await get_api_key())
    eff_key = key or ("sk-ignored" if provider == "vllm" else "")
    try:
        client = OpenAI(api_key=eff_key, base_url=base_url)  # type: ignore[call-arg]
        loop = asyncio.get_running_loop()

        def _list() -> list[str]:
            return [m.id for m in client.models.list().data]

        ids = await asyncio.wait_for(loop.run_in_executor(None, _list), timeout=15)
    except Exception:
        return [], [], []

    raw = list(ids)
    embed_keywords = [
        "embedding",
        "embed",
        "text-embedding",
        "e5",
        "bge",
        "jina",
        "gte",
        "nomic",
    ]
    emb = [m for m in ids if any(k in m.lower() for k in embed_keywords)]
    chat = [m for m in ids if m not in emb]
    return chat, emb, raw

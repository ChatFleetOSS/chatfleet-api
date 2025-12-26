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
        if key and OpenAI is not None:
            client = OpenAI(api_key=key, base_url=cfg.base_url)  # type: ignore[call-arg]
        else:
            client = _ensure_client_env_only()
    except Exception:
        client = _ensure_client_env_only()
    if client is None:
        return None

    loop = asyncio.get_running_loop()

    def _call() -> Tuple[str, int]:
        response = client.chat.completions.create(
            model=settings.chat_model,
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
        return await loop.run_in_executor(None, _call)
    except Exception:
        return None


async def test_chat_completion_provider(payload: LLMConfigTestRequest) -> tuple[bool, str | None]:
    if OpenAI is None:
        return False, "openai client not available"
    key = payload.api_key or os.getenv("OPENAI_API_KEY") or (await get_api_key())
    if not key:
        return False, "missing API key"
    try:
        client = OpenAI(api_key=key, base_url=payload.base_url)  # type: ignore[call-arg]
        # Prefer a cheap models call when available
        try:
            _ = client.models.list()
            return True, None
        except Exception:
            # Fallback to a 1-token completion
            _ = client.chat.completions.create(
                model=payload.chat_model or settings.chat_model,
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=1,
                temperature=0,
            )
            return True, None
    except Exception as exc:  # pragma: no cover
        return False, str(exc)

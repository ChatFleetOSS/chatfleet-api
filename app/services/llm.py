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

_client: OpenAI | None = None


def _ensure_client() -> OpenAI | None:
    global _client
    if _client is not None:
        return _client
    api_key = os.getenv("OPENAI_API_KEY")
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

    client = _ensure_client()
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

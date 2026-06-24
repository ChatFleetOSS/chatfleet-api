# generated-by: codex-agent 2025-02-17T12:40:00Z
"""
Large language model helpers for answer generation.

The implementation prefers OpenAI's Chat Completions API when credentials are
available, and falls back to `None` to let callers apply deterministic logic.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple, cast
from functools import partial

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - optional dependency
    OpenAI = None  # type: ignore

from app.core.config import settings
from app.core.corr_id import get_corr_id
from app.models.admin import LLMConfigTestRequest
from app.services.runtime_config import get_llm_config, get_api_key

_client_cache: dict[tuple[str | None, str | None], OpenAI] = {}
logger = logging.getLogger(__name__)
LLM_REQUEST_TIMEOUT = float(os.getenv("LLM_REQUEST_TIMEOUT", "60"))
_LLAMACPP_CHANNEL_PREFIX_RE = re.compile(r"(?is)^\s*<\|channel\>\s*([a-z0-9_-]+)\s*")
_LLAMACPP_CHANNEL_DELIMITER_RE = re.compile(r"(?is)<channel\|>")
_LLAMACPP_LEADING_MARKERS_RE = re.compile(
    r"(?is)^(?:\s*(?:<\|channel\>\s*[a-z0-9_-]+\s*|<channel\|>))+"
)


def _log_metric_event(event: str, payload: Dict[str, Any]) -> None:
    try:
        data = {"event": event, **payload}
        logger.info(
            "chatfleet.metrics %s",
            json.dumps(data, sort_keys=True, default=str),
            extra=payload,
        )
    except Exception:
        pass


class LLMProviderError(Exception):
    """Typed provider failure surfaced to chat routes with actionable messages."""

    def __init__(
        self,
        code: str,
        user_message: str,
        *,
        provider_message: str | None = None,
        status_code: int = 503,
    ) -> None:
        super().__init__(provider_message or user_message)
        self.code = code
        self.user_message = user_message
        self.provider_message = provider_message
        self.status_code = status_code


def _coerce_content_to_text(content: Any) -> str:
    """Normalize OpenAI-compatible content shapes into plain text."""

    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                if part.get("type") == "text":
                    parts.append(str(part.get("text") or ""))
                elif "text" in part:
                    parts.append(str(part.get("text") or ""))
        return "".join(parts).strip()
    return str(content).strip()


def _split_thinking_content(text: str) -> tuple[str, str]:
    if not text.lower().startswith("<think>"):
        return text, ""

    match = re.match(r"(?is)^<think>(.*?)</think>\s*(.*)$", text)
    if match:
        return match.group(2).strip(), match.group(1).strip()
    return "", text


def _split_llamacpp_channel_content(text: str) -> tuple[str, str]:
    """Remove llama.cpp channel markers from visible assistant content."""

    match = _LLAMACPP_CHANNEL_PREFIX_RE.match(text)
    if not match:
        return text, ""

    channel = match.group(1).lower()
    remaining = text[match.end() :]
    delimiter = _LLAMACPP_CHANNEL_DELIMITER_RE.search(remaining)
    reasoning = ""
    if delimiter:
        before_delimiter = remaining[: delimiter.start()].strip()
        if channel == "thought" and before_delimiter:
            reasoning = before_delimiter
        text = remaining[delimiter.end() :]
    else:
        if channel == "thought":
            reasoning = remaining.strip()
            text = ""
        else:
            text = remaining

    text = _LLAMACPP_LEADING_MARKERS_RE.sub("", text).strip()
    return text, reasoning


def _extract_message_text(choice: Any, response: Any) -> tuple[str, dict[str, Any]]:
    message = getattr(choice, "message", None)
    text = _coerce_content_to_text(getattr(message, "content", None))
    reasoning = _coerce_content_to_text(getattr(message, "reasoning_content", None))
    text, inline_reasoning = _split_thinking_content(text)
    if inline_reasoning:
        reasoning = f"{reasoning}\n{inline_reasoning}".strip()
    text, channel_reasoning = _split_llamacpp_channel_content(text)
    if channel_reasoning:
        reasoning = f"{reasoning}\n{channel_reasoning}".strip()
    finish_reason = getattr(choice, "finish_reason", None)
    usage = getattr(response, "usage", None)
    completion_tokens = getattr(usage, "completion_tokens", 0) or 0
    return text, {
        "finish_reason": finish_reason,
        "content_len": len(text),
        "reasoning_len": len(reasoning),
        "completion_tokens": completion_tokens,
    }


def _map_provider_exception(exc: Exception) -> LLMProviderError:
    name = exc.__class__.__name__
    msg = str(exc)
    low = msg.lower()
    if (
        "timeout" in low
        or "timed out" in low
        or name in {"APITimeoutError", "ReadTimeout"}
    ):
        return LLMProviderError(
            "LLM_TIMEOUT",
            "Le modèle local met trop de temps à répondre. Essayez une question plus ciblée, ou demandez à un administrateur d'augmenter le timeout, de réduire top_k, ou de choisir un modèle plus rapide.",
            provider_message=msg,
        )
    if "context" in low and ("exceed" in low or "too long" in low or "maximum" in low):
        return LLMProviderError(
            "LLM_CONTEXT_LIMIT",
            "La requête dépasse la fenêtre de contexte du modèle. Essayez une question plus ciblée ou réduisez l'historique; un administrateur peut aussi baisser top_k ou activer un budget de contexte plus strict.",
            provider_message=msg,
        )
    if "401" in msg or "unauthorized" in low or "api key" in low:
        return LLMProviderError(
            "LLM_AUTH_ERROR",
            "Le fournisseur LLM refuse l'authentification. Demandez à un administrateur de vérifier la clé API configurée.",
            provider_message=msg,
        )
    if "404" in msg or "not found" in low or "model" in low and "invalid" in low:
        return LLMProviderError(
            "LLM_INVALID_MODEL",
            "Le modèle configuré est introuvable ou invalide. Demandez à un administrateur de vérifier le nom du modèle et l'endpoint.",
            provider_message=msg,
        )
    if "connection" in low or "connect" in low or "refused" in low:
        return LLMProviderError(
            "LLM_PROVIDER_UNREACHABLE",
            "Le serveur LLM local est inaccessible. Demandez à un administrateur de vérifier que le serveur est démarré et que la base URL est correcte.",
            provider_message=msg,
        )
    return LLMProviderError(
        "LLM_PROVIDER_ERROR",
        "Le fournisseur LLM a retourné une erreur. Réessayez, ou contactez un administrateur avec l'identifiant de corrélation affiché.",
        provider_message=msg,
    )


def _get_chat_client(
    provider: str, base_url: str | None, key: str | None
) -> OpenAI | None:
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
        cfg = await get_llm_config()
    except Exception:
        cfg = None
    if cfg is None:
        return None

    key = os.getenv("OPENAI_API_KEY")
    try:
        key = key or (await get_api_key())
    except Exception:
        pass
    if cfg.provider == "vllm" and not cfg.base_url:
        return None
    base_url = None if cfg.provider == "openai" else cfg.base_url
    client = _get_chat_client(cfg.provider, base_url, key)
    if client is None:
        return None

    loop = asyncio.get_running_loop()
    corr_id = get_corr_id()

    def _call(model_name: str) -> Tuple[str, int]:
        started = time.perf_counter()
        response = client.chat.completions.create(
            model=model_name,
            messages=cast(Any, messages),
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=LLM_REQUEST_TIMEOUT,
        )
        if not response.choices:
            raise LLMProviderError(
                "LLM_EMPTY_RESPONSE",
                "Le modèle n'a retourné aucun choix de réponse. Réessayez ou demandez à un administrateur de vérifier le modèle configuré.",
            )
        choice = response.choices[0]
        text, metrics = _extract_message_text(choice, response)
        metrics["elapsed_ms"] = round((time.perf_counter() - started) * 1000.0, 3)
        metrics["model"] = model_name
        metrics["provider"] = cfg.provider
        metrics["corr_id"] = corr_id
        metrics["max_tokens"] = max_tokens
        metrics["temperature"] = temperature
        logger.info("LLM response metrics", extra=metrics)
        _log_metric_event("llm.response", metrics)
        if not text:
            if metrics["reasoning_len"] > 0:
                raise LLMProviderError(
                    "LLM_REASONING_WITHOUT_ANSWER",
                    "Le modèle a produit du raisonnement mais pas de réponse finale. Essayez une question plus courte; un administrateur peut augmenter le budget de sortie, réduire le contexte, ou configurer un modèle non-thinking.",
                    provider_message=(
                        f"finish_reason={metrics['finish_reason']} "
                        f"completion_tokens={metrics['completion_tokens']} "
                        f"reasoning_len={metrics['reasoning_len']}"
                    ),
                )
            raise LLMProviderError(
                "LLM_EMPTY_COMPLETION",
                "Le modèle a retourné une réponse vide. Réessayez, ou demandez à un administrateur de vérifier le modèle et le budget de sortie.",
                provider_message=(
                    f"finish_reason={metrics['finish_reason']} "
                    f"completion_tokens={metrics['completion_tokens']}"
                ),
            )
        completion_tokens = 0
        if getattr(response, "usage", None) is not None:
            completion_tokens = getattr(response.usage, "completion_tokens", 0) or 0
        if completion_tokens <= 0:
            # Fallback estimate when the API does not return usage statistics.
            completion_tokens = max(1, len(text.split()))
        return text, completion_tokens

    try:
        cfg = await get_llm_config()
        model_name = (cfg.chat_model or settings.chat_model).strip()
        if not model_name:

            def _pick_model() -> str:
                models = client.models.list().data
                if not models:
                    raise RuntimeError("No models returned by /v1/models")
                return models[0].id

            model_name = await loop.run_in_executor(None, _pick_model)
        return await loop.run_in_executor(None, partial(_call, model_name))
    except LLMProviderError:
        try:
            cfg = await get_llm_config()
            logger.exception(
                "LLM provider error (provider=%s base_url=%s model=%s)",
                cfg.provider,
                cfg.base_url,
                cfg.chat_model,
            )
        except Exception:
            logger.exception("LLM provider error")
        raise
    except Exception as exc:
        mapped = _map_provider_exception(exc)
        try:
            cfg = await get_llm_config()
            logger.exception(
                "LLM chat completion failed (provider=%s base_url=%s model=%s)",
                cfg.provider,
                cfg.base_url,
                cfg.chat_model,
            )
        except Exception:
            logger.exception("LLM chat completion failed")
        raise mapped


async def test_chat_completion_provider(
    payload: LLMConfigTestRequest,
) -> tuple[bool, str | None]:
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
        if (
            "Name or service not known" in msg
            or "Failed to establish a new connection" in msg
        ):
            return False, "CONNECTION_ERROR: check base_url/network"
        return False, f"PROVIDER_ERROR: {msg}"


async def discover_models(
    provider: str, base_url: Optional[str], api_key: Optional[str]
) -> tuple[list[str], list[str], list[str]]:
    """Return (chat_models, embedding_models, raw_models).

    Uses the OpenAI-compatible /v1/models when available. For vLLM, base_url is required.
    """
    if OpenAI is None:
        raise RuntimeError("CLIENT_MISSING: install openai package")
    if provider == "vllm" and not base_url:
        raise ValueError("CONFIG_ERROR: missing base_url for vLLM")

    key = api_key or os.getenv("OPENAI_API_KEY") or (await get_api_key())
    eff_key = key or ("sk-ignored" if provider == "vllm" else "")
    try:
        client = OpenAI(api_key=eff_key, base_url=base_url)  # type: ignore[call-arg]
        loop = asyncio.get_running_loop()

        def _list() -> list[str]:
            return [m.id for m in client.models.list().data]

        ids = await asyncio.wait_for(loop.run_in_executor(None, _list), timeout=15)
    except Exception as exc:
        msg = str(exc)
        if "401" in msg or "Unauthorized" in msg:
            raise RuntimeError("AUTH_ERROR: unauthorized") from exc
        if "timed out" in msg or "Timeout" in msg:
            raise RuntimeError("TIMEOUT: provider did not respond") from exc
        if "Not Found" in msg or "404" in msg:
            raise RuntimeError("INVALID_ENDPOINT: check base_url") from exc
        raise RuntimeError(f"PROVIDER_ERROR: {msg}") from exc

    if not ids:
        raise RuntimeError("NO_MODELS: provider returned empty list")

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

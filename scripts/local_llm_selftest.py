# generated-by: codex-agent 2026-01-26T00:00:00Z
"""
Quick self-test for a local OpenAI-compatible LLM endpoint and local embeddings.

What it does:
- Embeds sample texts with the same local model ChatFleet uses by default (BAAI/bge-m3).
- Calls the chat completion endpoint on a local OpenAI-compatible server (e.g., vLLM).
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Sequence

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

try:
    from openai import OpenAI
except Exception as exc:  # pragma: no cover - optional dependency
    print(f"[fatal] openai package missing: {exc}", file=sys.stderr)
    sys.exit(1)

try:
    from sentence_transformers import SentenceTransformer
except Exception as exc:  # pragma: no cover - optional dependency
    print(f"[fatal] sentence-transformers missing: {exc}", file=sys.stderr)
    sys.exit(1)


def embed_samples(model_name: str, texts: Sequence[str]) -> None:
    model = SentenceTransformer(model_name)
    vectors = model.encode(list(texts), normalize_embeddings=True, convert_to_numpy=True)
    print(f"[embeddings] model={model_name} shape={vectors.shape}")
    for text, vec in zip(texts, vectors):
        print(f"  - '{text}' -> dim={len(vec)} norm={float((vec**2).sum())**0.5:.4f}")


def pick_model(client: OpenAI, requested: str | None) -> str:
    if requested:
        return requested
    models = client.models.list().data
    if not models:
        raise RuntimeError("No models returned by /v1/models; cannot pick chat model")
    return models[0].id


def _extract_content(choice) -> str:
    """
    Normalize message content into a plain string.
    Supports string content and the content-part list shape.
    """
    content = getattr(choice.message, "content", "") or ""
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                parts.append(part.get("text", ""))
            elif isinstance(part, str):
                parts.append(part)
        content = "".join(parts)
    return content.strip()


def chat_once(client: OpenAI, model_name: str, prompt: str, timeout: float) -> None:
    resp = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=128,
        timeout=timeout,
    )
    choice = resp.choices[0] if resp.choices else None
    content = _extract_content(choice) if choice else ""
    print(
        f"[chat] model={model_name} tokens_out={getattr(resp.usage, 'completion_tokens', 'n/a')} "
        f"finish_reason={getattr(choice, 'finish_reason', '?') if choice else '?'}"
    )
    print(content or "<empty>")


def main() -> None:
    parser = argparse.ArgumentParser(description="Test local embeddings + local LLM chat.")
    parser.add_argument("--base-url", default=os.getenv("LLM_BASE_URL", "http://127.0.0.1:2242/v1"))
    parser.add_argument("--api-key", default=os.getenv("LLM_API_KEY", "sk-ignored"))
    parser.add_argument("--chat-model", default=os.getenv("LLM_CHAT_MODEL"))
    parser.add_argument("--embed-model", default=os.getenv("LLM_EMBED_MODEL", "BAAI/bge-m3"))
    parser.add_argument("--prompt", default="Give me one fun fact about local LLMs.")
    parser.add_argument("--timeout", type=float, default=float(os.getenv("LLM_TIMEOUT", "30")))
    args = parser.parse_args()

    samples = [
        "Local embeddings keep private data offline.",
        "vLLM serves OpenAI-compatible chat completions.",
    ]
    print("== Embedding sample texts with sentence-transformers ==")
    embed_samples(args.embed_model, samples)

    print("\n== Chat completion against local OpenAI-compatible endpoint ==")
    client = OpenAI(api_key=args.api_key, base_url=args.base_url, timeout=args.timeout)  # type: ignore[call-arg]
    model_name = pick_model(client, args.chat_model)
    print(f"[chat] using base_url={args.base_url} model={model_name}")
    chat_once(client, model_name, args.prompt, args.timeout)


if __name__ == "__main__":
    main()

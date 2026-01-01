import os
from typing import Optional

import httpx


BASE_URL = os.getenv("CHATFLEET_BASE_URL", "http://localhost:8000/api").rstrip("/")
ADMIN_EMAIL = os.getenv("ADMIN_EMAIL")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD")
EMAIL = os.getenv("EMAIL", ADMIN_EMAIL or "")
PASSWORD = os.getenv("PASSWORD", ADMIN_PASSWORD or "")
RAG_SLUG = os.getenv("RAG_SLUG")

VLLM_BASE_URL = os.getenv("VLLM_BASE_URL")
VLLM_API_KEY = os.getenv("VLLM_API_KEY")
CHAT_MODEL = os.getenv("CHAT_MODEL")
EMBED_PROVIDER = os.getenv("EMBED_PROVIDER", "local")
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
VERIFY_VLLM = os.getenv("VERIFY_VLLM", "1") == "1"
VLLM_TIMEOUT = float(os.getenv("VLLM_TIMEOUT", "60"))
VLLM_ALLOW_TIMEOUT = os.getenv("VLLM_ALLOW_TIMEOUT", "0") == "1"


def _login(client: httpx.Client, email: str, password: str) -> str:
    resp = client.post(f"{BASE_URL}/auth/login", json={"email": email, "password": password})
    resp.raise_for_status()
    return resp.json()["token"]


def _auth_headers(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


def _admin_configure_llm(client: httpx.Client, token: str) -> None:
    if not VLLM_BASE_URL:
        raise RuntimeError("VLLM_BASE_URL is required to configure vLLM chat.")
    payload = {
        "provider": "vllm",
        "base_url": VLLM_BASE_URL,
        "api_key": VLLM_API_KEY,
        "chat_model": CHAT_MODEL or "",
        "embed_provider": EMBED_PROVIDER,
        "embed_model": EMBED_MODEL,
    }
    resp = client.put(f"{BASE_URL}/admin/llm/config", json=payload, headers=_auth_headers(token))
    resp.raise_for_status()


def _admin_test_llm(client: httpx.Client, token: str) -> None:
    payload = {
        "provider": "vllm",
        "base_url": VLLM_BASE_URL,
        "api_key": VLLM_API_KEY,
        "chat_model": CHAT_MODEL or None,
        "embed_provider": EMBED_PROVIDER,
        "embed_model": EMBED_MODEL,
    }
    resp = client.post(f"{BASE_URL}/admin/llm/config/test", json=payload, headers=_auth_headers(token))
    resp.raise_for_status()
    print("Chat test:", resp.json())

    resp = client.post(f"{BASE_URL}/admin/llm/config/test-embed", json=payload, headers=_auth_headers(token))
    resp.raise_for_status()
    print("Embedding test:", resp.json())


def _vllm_headers() -> dict:
    if not VLLM_API_KEY:
        return {}
    return {"Authorization": f"Bearer {VLLM_API_KEY}"}


def _vllm_list_models(client: httpx.Client) -> list[str]:
    if not VLLM_BASE_URL:
        raise RuntimeError("VLLM_BASE_URL is required to verify vLLM.")
    resp = client.get(f"{VLLM_BASE_URL}/models", headers=_vllm_headers())
    resp.raise_for_status()
    data = resp.json()
    ids = [m["id"] for m in data.get("data", []) if "id" in m]
    if not ids:
        raise RuntimeError("No models returned by vLLM /v1/models.")
    return ids


def _vllm_chat_probe(client: httpx.Client, model_id: str) -> None:
    if not VLLM_BASE_URL:
        raise RuntimeError("VLLM_BASE_URL is required to verify vLLM.")
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": "ping"}],
        "max_tokens": 1,
    }
    try:
        resp = client.post(
            f"{VLLM_BASE_URL}/chat/completions",
            json=payload,
            headers=_vllm_headers(),
            timeout=VLLM_TIMEOUT,
        )
        resp.raise_for_status()
        print("vLLM chat OK.")
    except httpx.ReadTimeout:
        if VLLM_ALLOW_TIMEOUT:
            print("vLLM chat timed out; skipping (VLLM_ALLOW_TIMEOUT=1).")
            return
        raise


def _chat_probe(client: httpx.Client, token: str, rag_slug: str) -> None:
    payload = {
        "rag_slug": rag_slug,
        "messages": [{"role": "user", "content": "Say hello in one short sentence."}],
    }
    resp = client.post(f"{BASE_URL}/chat", json=payload, headers=_auth_headers(token))
    resp.raise_for_status()
    print("Chat response:", resp.json())


def main() -> None:
    if not EMAIL or not PASSWORD:
        raise RuntimeError("Set EMAIL/PASSWORD or ADMIN_EMAIL/ADMIN_PASSWORD.")

    with httpx.Client(timeout=30.0) as client:
        token = _login(client, EMAIL, PASSWORD)
        print("Login OK.")

        if ADMIN_EMAIL and ADMIN_PASSWORD:
            _admin_configure_llm(client, token)
            print("Admin config OK.")
            _admin_test_llm(client, token)

        if VERIFY_VLLM and VLLM_BASE_URL:
            models = _vllm_list_models(client)
            model_id = CHAT_MODEL or models[0]
            print("vLLM model:", model_id)
            _vllm_chat_probe(client, model_id)

        if RAG_SLUG:
            _chat_probe(client, token, RAG_SLUG)
        else:
            print("RAG_SLUG not set; skipping chat probe.")


if __name__ == "__main__":
    main()

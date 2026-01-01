from __future__ import annotations

import os
import sys
from typing import Any, Dict

import httpx


BASE_URL = os.getenv("CHATFLEET_BASE_URL", "http://localhost:8000/api").rstrip("/")
EMAIL = os.getenv("SMOKE_EMAIL", os.getenv("ADMIN_EMAIL", "admin@chatfleet.local"))
PASSWORD = os.getenv("SMOKE_PASSWORD", os.getenv("ADMIN_PASSWORD", "adminpass"))
RAG_SLUG = os.getenv("SMOKE_RAG_SLUG", "test3")
QUERY = os.getenv("SMOKE_QUERY", "qu'est ce que le syntec?")
TIMEOUT = float(os.getenv("SMOKE_TIMEOUT", "60"))


def log(step: str, msg: str) -> None:
    print(f"[{step}] {msg}")


def _auth_headers(token: str) -> Dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def login(client: httpx.Client) -> str:
    resp = client.post(f"{BASE_URL}/auth/login", json={"email": EMAIL, "password": PASSWORD})
    resp.raise_for_status()
    token = resp.json()["token"]
    log("login", "OK")
    return token


def get_llm_config(client: httpx.Client, token: str) -> None:
    resp = client.get(f"{BASE_URL}/admin/llm/config", headers=_auth_headers(token))
    resp.raise_for_status()
    cfg = resp.json()["config"]
    log("llm", f"provider={cfg['provider']} embed_provider={cfg.get('embed_provider')} chat_model={cfg.get('chat_model')}")


def chat_once(client: httpx.Client, token: str) -> None:
    payload = {
        "rag_slug": RAG_SLUG,
        "messages": [{"role": "user", "content": QUERY}],
    }
    resp = client.post(f"{BASE_URL}/chat", json=payload, headers=_auth_headers(token))
    resp.raise_for_status()
    data: Dict[str, Any] = resp.json()
    log("chat", f"corr_id={data.get('corr_id')} usage={data.get('usage')}")
    print("\n--- Answer ---\n")
    print(data.get("answer", "").strip())
    print("\n--- Citations ---\n")
    for c in data.get("citations", []):
        print(f"{c.get('filename')} pages={c.get('pages')} snippet={c.get('snippet')}")


def main() -> None:
    if not EMAIL or not PASSWORD:
        sys.exit("Set SMOKE_EMAIL/SMOKE_PASSWORD or ADMIN_EMAIL/ADMIN_PASSWORD.")
    with httpx.Client(timeout=TIMEOUT) as client:
        token = login(client)
        get_llm_config(client, token)
        chat_once(client, token)


if __name__ == "__main__":
    main()

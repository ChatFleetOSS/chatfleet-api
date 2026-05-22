# generated-by: codex-agent 2026-05-22T00:00:00Z
"""
Realistic RAG HTTP smoke test against a running ChatFleet API.

The script verifies the production-sensitive path: admin login, RAG creation,
document upload, background indexing, document inventory, index status, and a
chat probe. A configured LLM is required for a successful answer; without one,
the script still passes after proving ingestion/indexing and seeing an expected
LLM configuration error.

Usage:
  BASE_URL=http://localhost:8080/api \
  ADMIN_EMAIL=admin@chatfleet.local \
  ADMIN_PASSWORD=adminpass \
  python backend/scripts/rag_e2e_smoke.py

Set REQUIRE_CHAT_SUCCESS=1 when a real LLM provider is configured and the chat
probe must return an answer with citations.
"""

from __future__ import annotations

import os
import random
import string
import tempfile
import time
from pathlib import Path
from typing import Any

import httpx


BASE_URL = os.getenv("BASE_URL", "http://localhost:8080/api").rstrip("/")
ADMIN_EMAIL = os.getenv("ADMIN_EMAIL", "admin@chatfleet.local")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "adminpass")
RAG_SLUG = os.getenv("RAG_SLUG") or "regression-rag-" + "".join(
    random.choices(string.ascii_lowercase + string.digits, k=6)
)
POLL_SECONDS = float(os.getenv("POLL_SECONDS", "1.5"))
POLL_TIMEOUT = float(os.getenv("POLL_TIMEOUT", "120"))
REQUIRE_CHAT_SUCCESS = os.getenv("REQUIRE_CHAT_SUCCESS", "0") == "1"

SMOKE_TEXT = """ChatFleet regression smoke document.

Reference CF-REG-2026 states that existing client upgrades must preserve Mongo
secrets, non-empty Mongo URIs, Docker volumes, and RAG index paths.

The expected operational answer should mention preservation of existing client
configuration and cite this smoke document.
"""


def _auth_headers(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def _login(client: httpx.Client) -> str:
    resp = client.post(
        f"{BASE_URL}/auth/login",
        json={"email": ADMIN_EMAIL, "password": ADMIN_PASSWORD},
        timeout=15.0,
    )
    resp.raise_for_status()
    token = resp.json().get("token")
    if not token:
        raise RuntimeError("Login did not return a token")
    return token


def _ensure_rag(client: httpx.Client, token: str) -> None:
    payload = {
        "slug": RAG_SLUG,
        "name": RAG_SLUG,
        "description": "ChatFleet regression RAG smoke test",
    }
    resp = client.post(
        f"{BASE_URL}/rag",
        json=payload,
        headers=_auth_headers(token),
        timeout=15.0,
    )
    if resp.status_code in (200, 201, 409):
        return
    resp.raise_for_status()


def _make_txt() -> Path:
    tmp_dir = Path(tempfile.mkdtemp(prefix="chatfleet-rag-e2e-"))
    path = tmp_dir / "chatfleet-regression-smoke.txt"
    path.write_text(SMOKE_TEXT, encoding="utf-8")
    return path


def _upload_txt(client: httpx.Client, token: str, path: Path) -> str:
    with path.open("rb") as handle:
        resp = client.post(
            f"{BASE_URL}/rag/upload",
            files={"files": (path.name, handle, "text/plain")},
            data={"rag_slug": RAG_SLUG},
            headers=_auth_headers(token),
            timeout=60.0,
        )
    resp.raise_for_status()
    data = resp.json()
    skipped = data.get("skipped") or []
    if skipped:
        raise RuntimeError(f"Upload skipped files: {skipped}")
    job_id = data.get("job_id")
    if not job_id:
        raise RuntimeError(f"Upload did not return job_id: {data}")
    return job_id


def _poll_job(client: httpx.Client, token: str, job_id: str) -> dict[str, Any]:
    deadline = time.time() + POLL_TIMEOUT
    while time.time() < deadline:
        resp = client.get(
            f"{BASE_URL}/jobs/{job_id}",
            headers=_auth_headers(token),
            timeout=15.0,
        )
        resp.raise_for_status()
        data = resp.json()
        status = data.get("status")
        print(f"[job] status={status} phase={data.get('phase')} progress={data.get('progress')}")
        if status in {"done", "error"}:
            return data
        time.sleep(POLL_SECONDS)
    raise TimeoutError(f"Job {job_id} did not finish within {POLL_TIMEOUT}s")


def _assert_docs_indexed(client: httpx.Client, token: str, filename: str) -> None:
    resp = client.get(
        f"{BASE_URL}/rag/docs",
        params={"rag_slug": RAG_SLUG},
        headers=_auth_headers(token),
        timeout=15.0,
    )
    resp.raise_for_status()
    docs = resp.json().get("docs") or []
    doc = next((item for item in docs if item.get("filename") == filename), None)
    if not doc:
        raise RuntimeError(f"{filename} not found in /rag/docs: {docs}")
    if doc.get("status") != "indexed":
        raise RuntimeError(f"{filename} status is not indexed: {doc}")
    if (doc.get("chunk_count") or 0) <= 0:
        raise RuntimeError(f"{filename} indexed with no chunks: {doc}")


def _assert_index_status(client: httpx.Client, token: str) -> None:
    resp = client.get(
        f"{BASE_URL}/rag/index/status",
        params={"rag_slug": RAG_SLUG},
        headers=_auth_headers(token),
        timeout=15.0,
    )
    resp.raise_for_status()
    data = resp.json()
    print("[index]", data)
    chunks = data.get("total_chunks") or data.get("chunk_count") or 0
    if int(chunks) <= 0:
        raise RuntimeError(f"Index status reports no chunks: {data}")


def _chat_probe(client: httpx.Client, token: str) -> None:
    payload = {
        "rag_slug": RAG_SLUG,
        "messages": [
            {
                "role": "user",
                "content": "What does reference CF-REG-2026 require during upgrades?",
            }
        ],
    }
    resp = client.post(
        f"{BASE_URL}/chat",
        json=payload,
        headers=_auth_headers(token),
        timeout=60.0,
    )
    if resp.status_code == 200:
        data = resp.json()
        answer = data.get("answer") or ""
        citations = data.get("citations") or []
        if not answer:
            raise RuntimeError(f"Chat returned empty answer: {data}")
        if not citations:
            raise RuntimeError(f"Chat returned no citations: {data}")
        print("[chat] success", {"corr_id": data.get("corr_id"), "citations": len(citations)})
        return

    data = resp.json()
    code = ((data.get("detail") or {}).get("error") or {}).get("code")
    if REQUIRE_CHAT_SUCCESS:
        raise RuntimeError(f"Chat failed but REQUIRE_CHAT_SUCCESS=1: status={resp.status_code} body={data}")
    if code not in {"LLM_NOT_CONFIGURED", "LLM_UNAVAILABLE", "NO_CONTEXT"}:
        raise RuntimeError(f"Unexpected chat failure: status={resp.status_code} body={data}")
    print("[chat] expected non-success", {"status": resp.status_code, "code": code})


def main() -> int:
    with httpx.Client() as client:
        print(f"[health] {BASE_URL}")
        health = client.get(f"{BASE_URL}/health", timeout=15.0)
        health.raise_for_status()

        print(f"[login] {ADMIN_EMAIL}")
        token = _login(client)

        print(f"[rag] ensure {RAG_SLUG}")
        _ensure_rag(client, token)

        path = _make_txt()
        print(f"[upload] {path.name}")
        job_id = _upload_txt(client, token, path)
        result = _poll_job(client, token, job_id)
        if result.get("status") != "done":
            raise RuntimeError(f"Index job failed: {result}")

        print("[docs] verify indexed")
        _assert_docs_indexed(client, token, path.name)
        _assert_index_status(client, token)
        _chat_probe(client, token)

    print("[ok] RAG E2E smoke succeeded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

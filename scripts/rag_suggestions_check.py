# generated-by: codex-agent 2026-02-01T23:30:00Z
"""
Smoke test: create/upload a RAG and verify suggestions are generated (LLM-based).

Usage (env):
  BASE_URL=http://localhost:8000/api \
  ADMIN_EMAIL=admin@chatfleet.local \
  ADMIN_PASSWORD=adminpass \
  RAG_SLUG=suggestions-demo \
  DOC_PATH=var/uploads/code_odt/oodoc_guide.odt \
  python scripts/rag_suggestions_check.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import httpx

BASE_URL = os.getenv("BASE_URL", "http://localhost:8000/api").rstrip("/")
ADMIN_EMAIL = os.getenv("ADMIN_EMAIL", "admin@chatfleet.local")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "adminpass")
RAG_SLUG = os.getenv("RAG_SLUG", "suggestions-demo")
DOC_PATH = Path(os.getenv("DOC_PATH", "var/uploads/code_odt/oodoc_guide.odt"))
POLL_SECONDS = float(os.getenv("POLL_SECONDS", "1.5"))
POLL_TIMEOUT = float(os.getenv("POLL_TIMEOUT", "120"))


def _login(client: httpx.Client) -> str:
    resp = client.post(f"{BASE_URL}/auth/login", json={"email": ADMIN_EMAIL, "password": ADMIN_PASSWORD})
    resp.raise_for_status()
    token = resp.json().get("token")
    if not token:
        raise RuntimeError("Login did not return a token")
    return token


def _auth(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


def _ensure_rag(client: httpx.Client, token: str) -> None:
    payload = {
        "slug": RAG_SLUG,
        "name": RAG_SLUG,
        "description": "Suggestion generation smoke test",
        "visibility": "private",
    }
    resp = client.post(f"{BASE_URL}/rag", json=payload, headers=_auth(token))
    if resp.status_code not in (200, 201, 409):
        resp.raise_for_status()


def _upload_doc(client: httpx.Client, token: str, path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("rb") as f:
        files = {"files": (path.name, f, "application/octet-stream")}
        data = {"rag_slug": RAG_SLUG}
        resp = client.post(f"{BASE_URL}/rag/upload", files=files, data=data, headers=_auth(token))
    resp.raise_for_status()
    job_id = resp.json().get("job_id")
    if not job_id:
        raise RuntimeError("Upload did not return job_id")
    return job_id


def _poll_job_for_suggestions(client: httpx.Client, token: str, job_id: str) -> dict:
    deadline = time.time() + POLL_TIMEOUT
    while time.time() < deadline:
        resp = client.get(f"{BASE_URL}/jobs/{job_id}", headers=_auth(token))
        resp.raise_for_status()
        data = resp.json()
        status = data.get("status")
        suggestions_ready = data.get("suggestions_ready", False)
        if status == "done" and suggestions_ready:
            return data
        if status == "error":
            raise RuntimeError(f"Job failed: {data}")
        time.sleep(POLL_SECONDS)
    raise TimeoutError(f"Job {job_id} did not finish with suggestions within {POLL_TIMEOUT}s")


def _fetch_suggestions(client: httpx.Client, token: str) -> list[str]:
    resp = client.get(f"{BASE_URL}/rag/list", headers=_auth(token))
    resp.raise_for_status()
    items = resp.json().get("items", [])
    for item in items:
        if item.get("slug") == RAG_SLUG:
            return item.get("suggestions") or []
    return []


def main() -> int:
    with httpx.Client(timeout=30.0) as client:
        print(f"[login] {ADMIN_EMAIL}")
        token = _login(client)
        print("[ok] authenticated")

        print(f"[rag] ensure slug '{RAG_SLUG}'")
        _ensure_rag(client, token)

        print(f"[upload] {DOC_PATH}")
        job_id = _upload_doc(client, token, DOC_PATH)
        print(f"[job] {job_id} -> polling for suggestions")
        job_data = _poll_job_for_suggestions(client, token, job_id)
        print("[job] done with suggestions flag =", job_data.get("suggestions_ready"))

        print("[verify] fetch suggestions from rag list")
        suggestions = _fetch_suggestions(client, token)
        if not suggestions:
            raise RuntimeError("No suggestions returned in rag list")
        preview = "; ".join(suggestions[:3])
        print(f"[ok] suggestions count={len(suggestions)} preview={preview}")

    print("[ok] suggestion generation flow succeeded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

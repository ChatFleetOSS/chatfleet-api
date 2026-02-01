# generated-by: codex-agent 2026-02-01T00:00:00Z
"""
Create and verify a public RAG end-to-end, including unauthenticated chat.

Usage (env):
  BASE_URL=http://localhost:8000/api \
  ADMIN_EMAIL=admin@chatfleet.local \
  ADMIN_PASSWORD=adminpass \
  RAG_SLUG=odt-public \
  ODT_PATH=var/uploads/code_odt/oodoc_guide.odt \
  python scripts/public_rag_check.py
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
RAG_SLUG = os.getenv("RAG_SLUG", "odt-public")
ODT_PATH = Path(os.getenv("ODT_PATH", "var/uploads/code_odt/oodoc_guide.odt"))
POLL_SECONDS = float(os.getenv("POLL_SECONDS", "1.5"))
POLL_TIMEOUT = float(os.getenv("POLL_TIMEOUT", "90"))


def _login(client: httpx.Client) -> str:
    resp = client.post(f"{BASE_URL}/auth/login", json={"email": ADMIN_EMAIL, "password": ADMIN_PASSWORD})
    resp.raise_for_status()
    token = resp.json().get("token")
    if not token:
        raise RuntimeError("Login did not return a token")
    return token


def _auth_headers(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


def _ensure_public_rag(client: httpx.Client, token: str) -> None:
    payload = {
        "slug": RAG_SLUG,
        "name": RAG_SLUG,
        "description": "Public RAG smoke test",
        "visibility": "public",
    }
    resp = client.post(f"{BASE_URL}/rag", json=payload, headers=_auth_headers(token))
    if resp.status_code not in (200, 201, 409):
        resp.raise_for_status()


def _upload_odt(client: httpx.Client, token: str, path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("rb") as f:
        files = {"files": (path.name, f, "application/vnd.oasis.opendocument.text")}
        data = {"rag_slug": RAG_SLUG}
        resp = client.post(f"{BASE_URL}/rag/upload", files=files, data=data, headers=_auth_headers(token))
    resp.raise_for_status()
    job_id = resp.json().get("job_id")
    if not job_id:
        raise RuntimeError("Upload did not return a job_id")
    return job_id


def _poll_job(client: httpx.Client, token: str, job_id: str) -> None:
    deadline = time.time() + POLL_TIMEOUT
    while time.time() < deadline:
        resp = client.get(f"{BASE_URL}/jobs/{job_id}", headers=_auth_headers(token))
        resp.raise_for_status()
        data = resp.json()
        status = data.get("status")
        if status == "done":
            return
        if status == "error":
            raise RuntimeError(f"Job failed: {data}")
        time.sleep(POLL_SECONDS)
    raise TimeoutError(f"Job {job_id} did not finish within {POLL_TIMEOUT}s")


def _public_list_has_slug(client: httpx.Client) -> bool:
    resp = client.get(f"{BASE_URL}/public/rag/list")
    resp.raise_for_status()
    items = resp.json().get("items", [])
    return any(item.get("slug") == RAG_SLUG for item in items)


def _public_chat(client: httpx.Client, question: str) -> str:
    resp = client.post(
        f"{BASE_URL}/public/chat",
        json={"rag_slug": RAG_SLUG, "messages": [{"role": "user", "content": question}]},
    )
    resp.raise_for_status()
    data = resp.json()
    return data.get("answer") or ""


def main() -> int:
    with httpx.Client(timeout=30.0) as client:
        print(f"[login] {ADMIN_EMAIL}")
        token = _login(client)
        print("[ok] authenticated")

        print(f"[rag] ensure public slug '{RAG_SLUG}'")
        _ensure_public_rag(client, token)

        print(f"[upload] {ODT_PATH}")
        job_id = _upload_odt(client, token, ODT_PATH)
        print(f"[job] {job_id}")
        _poll_job(client, token, job_id)
        print("[job] done")

        print("[public list] checking visibility")
        if not _public_list_has_slug(client):
            raise RuntimeError("Public rag not visible in /public/rag/list")
        print("[public list] ok")

        print("[chat] querying anonymously")
        answer = _public_chat(client, "How do you run a Perl script from the command line?")
        preview = (answer[:240] + "...") if len(answer) > 240 else answer
        if not answer:
            raise RuntimeError("Public chat returned empty answer")
        print("[chat] ok ->", preview)

    print("[ok] public RAG flow succeeded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

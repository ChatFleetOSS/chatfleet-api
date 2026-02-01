# generated-by: codex-agent 2026-02-01T00:00:00Z
"""
End-to-end ingestion probe for ODT documents.

What it does:
- Logs in with admin credentials.
-,Creates (or reuses) a RAG slug.
- Generates a tiny ODT file on disk.
- Uploads it via /api/rag/upload.
- Polls /api/jobs/{id} until the index build finishes.
- Confirms the document reaches status=indexed via /api/rag/docs.

Usage (env):
  BASE_URL=http://localhost:8000/api \
  ADMIN_EMAIL=admin@chatfleet.local \
  ADMIN_PASSWORD=adminpass \
  RAG_SLUG=odt-e2e \
  python scripts/odt_ingest_check.py
"""

from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path

import httpx
from odf.opendocument import OpenDocumentText
from odf.table import Table, TableRow, TableCell
from odf.text import P

BASE_URL = os.getenv("BASE_URL", "http://localhost:8000/api").rstrip("/")
ADMIN_EMAIL = os.getenv("ADMIN_EMAIL", "admin@chatfleet.local")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "adminpass")
RAG_SLUG = os.getenv("RAG_SLUG", "odt-e2e")
POLL_SECONDS = float(os.getenv("POLL_SECONDS", "1.5"))
POLL_TIMEOUT = float(os.getenv("POLL_TIMEOUT", "90"))


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


def _auth_headers(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


def _ensure_rag(client: httpx.Client, token: str) -> None:
    payload = {"slug": RAG_SLUG, "name": RAG_SLUG, "description": "ODT ingest probe"}
    resp = client.post(f"{BASE_URL}/rag", json=payload, headers=_auth_headers(token), timeout=15.0)
    if resp.status_code in (200, 201, 409):
        return
    resp.raise_for_status()


def _make_odt_file() -> Path:
    doc = OpenDocumentText()
    doc.text.addElement(P(text="ChatFleet ODT ingest test document."))

    table = Table(name="Sample")
    row = TableRow()
    cell = TableCell()
    cell.addElement(P(text="Row 1, Col 1"))
    row.addElement(cell)
    table.addElement(row)
    doc.text.addElement(table)

    tmp_dir = Path(tempfile.mkdtemp(prefix="odt-ingest-"))
    path = tmp_dir / "odt-ingest-check.odt"
    doc.save(str(path))
    return path


def _upload_odt(client: httpx.Client, token: str, odt_path: Path) -> str:
    with odt_path.open("rb") as f:
        files = {"files": (odt_path.name, f, "application/vnd.oasis.opendocument.text")}
        data = {"rag_slug": RAG_SLUG}
        resp = client.post(
            f"{BASE_URL}/rag/upload",
            files=files,
            data=data,
            headers=_auth_headers(token),
            timeout=60.0,
        )
    resp.raise_for_status()
    job_id = resp.json().get("job_id")
    if not job_id:
        raise RuntimeError("Upload did not return a job_id")
    return job_id


def _poll_job(client: httpx.Client, token: str, job_id: str) -> dict:
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
        if status in {"done", "error"}:
            return data
        time.sleep(POLL_SECONDS)
    raise TimeoutError(f"Job {job_id} did not finish within {POLL_TIMEOUT}s")


def _assert_indexed(client: httpx.Client, token: str) -> None:
    resp = client.get(
        f"{BASE_URL}/rag/docs",
        params={"rag_slug": RAG_SLUG},
        headers=_auth_headers(token),
        timeout=15.0,
    )
    resp.raise_for_status()
    docs = resp.json().get("docs", [])
    doc = next((d for d in docs if d.get("filename", "").endswith(".odt")), None)
    if not doc:
        raise RuntimeError("Uploaded ODT document not found in rag/docs")
    if doc.get("status") != "indexed":
        raise RuntimeError(f"Document status not indexed: {doc.get('status')}")
    if (doc.get("chunk_count") or 0) <= 0:
        raise RuntimeError("Indexed document has zero chunks")


def main() -> int:
    with httpx.Client() as client:
        print(f"[login] {ADMIN_EMAIL} -> {BASE_URL}")
        token = _login(client)
        print("[ok] authenticated")

        print(f"[rag] ensure slug '{RAG_SLUG}' exists")
        _ensure_rag(client, token)

        odt_path = _make_odt_file()
        print(f"[upload] sending {odt_path.name}")
        job_id = _upload_odt(client, token, odt_path)
        print(f"[job] {job_id}")

        result = _poll_job(client, token, job_id)
        print(f"[job] status={result.get('status')} progress={result.get('progress')}")
        if result.get("status") != "done":
            raise RuntimeError(f"Job finished with status {result.get('status')}: {result}")

        print("[verify] checking rag/docs")
        _assert_indexed(client, token)

        print("[ok] ODT ingestion flow succeeded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

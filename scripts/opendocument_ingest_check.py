# generated-by: codex-agent 2026-05-04T00:00:00Z
"""
End-to-end ingestion smoke check for OpenDocument files.

What it does:
- Logs in with admin credentials.
- Creates or reuses a RAG slug.
- Generates tiny ODT, ODS, and ODP files with odfpy.
- Uploads all files through /api/rag/upload.
- Polls /api/jobs/{id} until the index build finishes.
- Confirms each document reaches status=indexed with chunks via /api/rag/docs.

Usage:
  BASE_URL=http://localhost:8000/api \
  ADMIN_EMAIL=admin@chatfleet.local \
  ADMIN_PASSWORD=adminpass \
  RAG_SLUG=opendocument-e2e \
  python scripts/opendocument_ingest_check.py
"""

from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path

import httpx
from odf.draw import Frame, Page, TextBox
from odf.opendocument import (
    OpenDocumentPresentation,
    OpenDocumentSpreadsheet,
    OpenDocumentText,
)
from odf.style import MasterPage, PageLayout, PageLayoutProperties
from odf.table import Table, TableCell, TableRow
from odf.text import P

BASE_URL = os.getenv("BASE_URL", "http://localhost:8000/api").rstrip("/")
ADMIN_EMAIL = os.getenv("ADMIN_EMAIL", "admin@chatfleet.local")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "adminpass")
RAG_SLUG = os.getenv("RAG_SLUG", "opendocument-e2e")
POLL_SECONDS = float(os.getenv("POLL_SECONDS", "1.5"))
POLL_TIMEOUT = float(os.getenv("POLL_TIMEOUT", "120"))

MIME_BY_EXT = {
    ".odt": "application/vnd.oasis.opendocument.text",
    ".ods": "application/vnd.oasis.opendocument.spreadsheet",
    ".odp": "application/vnd.oasis.opendocument.presentation",
}


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
    payload = {
        "slug": RAG_SLUG,
        "name": RAG_SLUG,
        "description": "OpenDocument ingest smoke check",
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


def _make_odt(path: Path) -> None:
    doc = OpenDocumentText()
    doc.text.addElement(P(text="ChatFleet ODT smoke document."))
    table = Table(name="ODT Smoke")
    row = TableRow()
    cell = TableCell()
    cell.addElement(P(text="ODT table smoke cell"))
    row.addElement(cell)
    table.addElement(row)
    doc.text.addElement(table)
    doc.save(str(path))


def _make_ods(path: Path) -> None:
    doc = OpenDocumentSpreadsheet()
    table = Table(name="Sheet1")
    row = TableRow()
    for text in ("ChatFleet ODS smoke cell", "ODS second cell"):
        cell = TableCell()
        cell.addElement(P(text=text))
        row.addElement(cell)
    table.addElement(row)
    doc.spreadsheet.addElement(table)
    doc.save(str(path))


def _make_odp(path: Path) -> None:
    doc = OpenDocumentPresentation()
    layout = PageLayout(name="pm1")
    layout.addElement(
        PageLayoutProperties(
            margin="0cm",
            pagewidth="28cm",
            pageheight="21cm",
            printorientation="landscape",
        )
    )
    doc.automaticstyles.addElement(layout)
    master = MasterPage(name="Default", pagelayoutname=layout)
    doc.masterstyles.addElement(master)

    slide = Page(masterpagename=master)
    frame = Frame(width="20cm", height="5cm", x="1cm", y="1cm")
    box = TextBox()
    box.addElement(P(text="ChatFleet ODP smoke slide."))
    frame.addElement(box)
    slide.addElement(frame)
    doc.presentation.addElement(slide)
    doc.save(str(path))


def _make_opendocument_files() -> list[Path]:
    tmp_dir = Path(tempfile.mkdtemp(prefix="opendocument-ingest-"))
    outputs = [
        tmp_dir / "opendocument-ingest-check.odt",
        tmp_dir / "opendocument-ingest-check.ods",
        tmp_dir / "opendocument-ingest-check.odp",
    ]
    _make_odt(outputs[0])
    _make_ods(outputs[1])
    _make_odp(outputs[2])
    return outputs


def _upload_files(client: httpx.Client, token: str, paths: list[Path]) -> str:
    opened = []
    try:
        files = []
        for path in paths:
            handle = path.open("rb")
            opened.append(handle)
            ext = path.suffix.lower()
            files.append(("files", (path.name, handle, MIME_BY_EXT[ext])))
        resp = client.post(
            f"{BASE_URL}/rag/upload",
            files=files,
            data={"rag_slug": RAG_SLUG},
            headers=_auth_headers(token),
            timeout=60.0,
        )
    finally:
        for handle in opened:
            handle.close()

    resp.raise_for_status()
    data = resp.json()
    skipped = data.get("skipped") or []
    if skipped:
        raise RuntimeError(f"Upload skipped files: {skipped}")
    job_id = data.get("job_id")
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
        print(
            f"[job] status={status} phase={data.get('phase')} progress={data.get('progress')}"
        )
        if status in {"done", "error"}:
            return data
        time.sleep(POLL_SECONDS)
    raise TimeoutError(f"Job {job_id} did not finish within {POLL_TIMEOUT}s")


def _assert_indexed(client: httpx.Client, token: str, paths: list[Path]) -> None:
    resp = client.get(
        f"{BASE_URL}/rag/docs",
        params={"rag_slug": RAG_SLUG},
        headers=_auth_headers(token),
        timeout=15.0,
    )
    resp.raise_for_status()
    docs = resp.json().get("docs", [])
    by_name = {doc.get("filename"): doc for doc in docs}

    for path in paths:
        doc = by_name.get(path.name)
        if not doc:
            raise RuntimeError(f"{path.name} not found in rag/docs")
        if doc.get("status") != "indexed":
            raise RuntimeError(f"{path.name} status not indexed: {doc.get('status')}")
        if (doc.get("chunk_count") or 0) <= 0:
            raise RuntimeError(f"{path.name} indexed with zero chunks")


def main() -> int:
    with httpx.Client() as client:
        print(f"[login] {ADMIN_EMAIL} -> {BASE_URL}")
        token = _login(client)
        print("[ok] authenticated")

        print(f"[rag] ensure slug '{RAG_SLUG}' exists")
        _ensure_rag(client, token)

        paths = _make_opendocument_files()
        print("[upload] " + ", ".join(path.name for path in paths))
        job_id = _upload_files(client, token, paths)
        print(f"[job] {job_id}")

        result = _poll_job(client, token, job_id)
        if result.get("status") != "done":
            raise RuntimeError(f"Job finished with status {result.get('status')}: {result}")

        print("[verify] checking rag/docs")
        _assert_indexed(client, token, paths)

        print("[ok] OpenDocument ingestion flow succeeded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

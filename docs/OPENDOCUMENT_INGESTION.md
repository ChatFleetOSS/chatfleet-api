# OpenDocument Ingestion

This document defines the production contract for ChatFleet RAG ingestion of
OpenDocument files.

## Supported Formats

The admin RAG upload flow supports:

| Extension | MIME type | Extraction model |
| --- | --- | --- |
| `.odt` | `application/vnd.oasis.opendocument.text` | Paragraphs and tables as sequential text units |
| `.ods` | `application/vnd.oasis.opendocument.spreadsheet` | Sheets, rows, and cells as sequential text units |
| `.odp` | `application/vnd.oasis.opendocument.presentation` | Slides, headings, paragraphs, and text boxes |

Existing PDF, DOCX, and TXT ingestion remains supported and must not regress.

## API Contract

OpenDocument uploads use the existing RAG ingestion endpoints:

- `POST /api/rag/upload`
- `GET /api/jobs/{job_id}`
- `GET /api/rag/docs?rag_slug=<slug>`
- `POST /api/rag/rebuild`
- `POST /api/rag/reset`

The frontend must call these through the shared API client. In local dev, browser
requests go to `/backend-api/*`, and Next.js rewrites those calls to
`http://localhost:8000/api/*`.

## Security Requirements

OpenDocument files are ZIP containers. Ingestion must validate the package before
persisting it as an accepted document:

- extension and MIME type must match the allowlist;
- the upload is stored under a generated server filename;
- upload size is enforced while streaming;
- the ZIP must be readable and pass `testzip()`;
- `mimetype`, `content.xml`, and `META-INF/manifest.xml` must exist;
- the internal `mimetype` value must match the uploaded extension;
- fake OpenDocument files, corrupt ZIPs, and fake PDFs must be rejected.

The backend must not shell out to LibreOffice/OpenOffice for extraction.

## User Stories

1. As an admin, I can upload `.odt` policy documents and have them indexed for chat.
2. As an admin, I can upload `.ods` spreadsheets and preserve sheet/row context in extracted text.
3. As an admin, I can upload `.odp` presentations and preserve slide context in extracted text.
4. As an admin, I get a rejected upload for fake or corrupt OpenDocument files.
5. As a user, existing PDF/DOCX/TXT ingestion keeps working after OpenDocument support is added.
6. As an operator, I can validate the feature with local unit tests, contract tests, and an API smoke check.

## Validation Matrix

Required local checks before release:

- `ruff check .`
- `python -m compileall -q app main.py tests scripts`
- `mypy --ignore-missing-imports --no-strict-optional app`
- `python -m unittest tests.test_opendocument_ingestion_unit tests.test_suggestions_unit -v`
- `npm run pact:consumer`
- `npm run pact:provider` with backend started using `CHATFLEET_FAKE_CHAT_MODE=1`

Frontend checks:

- `npm run build`
- `npx tsc --noEmit`

End-to-end smoke check through the frontend proxy:

```bash
cd backend
source .venv/bin/activate
BASE_URL=http://localhost:3000/backend-api \
ADMIN_EMAIL=admin@chatfleet.local \
ADMIN_PASSWORD=adminpass \
RAG_SLUG=fe-proxy-odf-test \
python scripts/opendocument_ingest_check.py
```

## CI And Release

Main branch CI must keep existing ingestion checks green. Pact remains focused on
stable API shape for multipart uploads and chat/delete contracts; OpenDocument
container correctness is covered by unit tests and the smoke script because Pact
multipart fixtures are not a reliable representation of binary ODF containers.

For installer availability, backend and frontend changes must be pushed to their
respective `main` branches, then release tags must be pushed so GHCR publishes
new `:latest` images. The infra `install.sh` pulled by curl will then install the
new images by default.

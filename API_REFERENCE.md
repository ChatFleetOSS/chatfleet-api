// generated-by: codex-agent 2025-02-17T12:05:00Z
# API Reference

This guide enumerates every `/api` endpoint exposed by the ChatFleet backend and explains how the frontend should call them. All endpoints return JSON with a `corr_id` for traceability, and errors follow a shared envelope.

## Conventions
- **Base URL:** `https://<host>/api`
- **Auth:** Bearer token obtained from `POST /api/auth/login`. Attach with `Authorization: Bearer <jwt>`.
- **Correlation IDs:** Success bodies include `corr_id`. Errors surface as `{"detail": {"error": {"code", "message"}, "corr_id"}}`.
- **Jobs:** Long-running tasks return `202 Accepted` with a `job_id`. Poll `GET /api/jobs/{job_id}` until `status` is `done` or `error`.
- **RAG slugs:** Lowercase, hyphenated identifiers matching Mongo `rags.slug`. Users must list in their `rags` array to chat.

## Auth
### `POST /api/auth/register`
- **Access:** Public.
- **Body (JSON):**
  ```json
  {"email": "user@example.com", "name": "Ada Lovelace", "password": "hunter42"}
  ```
- **Response 201:** 
  ```json
  {
    "token": "<jwt>",
    "user": {"_id": "65f...", "email": "...", "name": "...", "role": "user", "rags": [], "created_at": "...", "updated_at": "..."},
    "corr_id": "..."
  }
  ```

Notes:
- If an admin promotion intent exists for this email (created by the installer), the backend upgrades the role to `admin` during registration and the returned token/user reflect the admin role immediately.

### `POST /api/auth/login`
- **Access:** Public.
- **Body:** Same shape as register without `name`.
- **Response 200:** Auth token + user payload identical to register.

Notes:
- If an admin promotion intent exists for the email, the backend redeems it at login before issuing the token so the user sees the admin UI right away.

### `GET /api/auth/me`
- **Access:** Bearer required.
- **Response 200:** `UserPublic` (`_id`, `email`, `name`, `role`, `rags`, timestamps).

## Users (Admin)
### `GET /api/admin/users`
- **Access:** Admin only.
- **Query:** `limit` (1-200, default 50), optional `cursor` for pagination.
- **Response 200:** 
  ```json
  {"items": [UserPublic...], "next_cursor": "65f...", "corr_id": "..."}
  ```

### `POST /api/admin/users`
- **Access:** Admin only.
- **Body:** 
  ```json
  {"email": "pm@example.com", "name": "PM", "role": "admin", "rags": ["hr-docs"]}
  ```
- **Response 201:** Created `UserPublic`. Backend auto-generates a secure password and logs the event; surface it to the human admin out of band.

### RAG Membership APIs
- **`GET /api/rag/users?rag_slug=<slug>`** (admin): returns `{"rag_slug": "...", "users": [{"_id": "...", "email": "...", "name": "...", "role": "user"}], "corr_id": "..."}`.
- **`POST /api/rag/users/add`** (admin): JSON body must contain `rag_slug` and exactly one of `user_id` or `email`. Response echoes `{rag_slug, user_id, corr_id}`.
- **`POST /api/rag/users/remove`** (admin): Same payload contract; removes access and updates the user’s `rags` list.

## RAG Directory
### `GET /api/rag/list`
- **Access:** Any authenticated user. Non-admins see their shared slugs **plus** public RAGs.
- **Query:** `limit`, `cursor` (optional).
- **Response 200:** 
  ```json
  {"items": [{"slug": "hr-docs", "name": "HR Docs", "description": "...", "chunks": 1200, "last_updated": "...", "visibility": "private"}], "next_cursor": null, "corr_id": "..."}
  ```

### `GET /api/admin/rag/list`
- **Access:** Admin.
- **Purpose:** Same response as `/rag/list` but unfiltered.

### `GET /api/public/rag/list`
- **Access:** Public (no auth).
- **Response:** Same shape as `/api/rag/list` but only includes `visibility="public"` entries.

### `GET /api/public/rag/docs?rag_slug=<slug>`
- **Access:** Public if the RAG is marked `public`.
- **Response:** Document metadata for the public RAG (no user info).

## RAG Lifecycle (Admin)
### `POST /api/rag`
- **Access:** Admin.
- **Body:**
  ```json
  {"slug": "new-rag", "name": "New Knowledge Base", "description": "Short summary", "visibility": "private"}
  ```
- **Response 201:**
  ```json
  {
    "rag": {"slug": "new-rag", "name": "New Knowledge Base", "description": "Short summary", "chunks": 0, "last_updated": "...", "visibility": "private"},
    "corr_id": "..."
  }
  ```
- **Side effects:** Creates an empty RAG document with no users or files and logs `rag.create`. Follow with `/api/rag/users/add` to grant access and `/api/rag/upload` to ingest PDFs, DOCX, ODT, or TXT files.

### `POST /api/rag/upload`
- **Access:** Admin.
- **Content-Type:** `multipart/form-data` with fields:
  - `rag_slug`: string
  - `files`: one or more PDF/DOCX/ODT/TXT uploads
  - `splitter_opts` (optional JSON string; reserved for custom chunking)
- **Response 202:** 
  ```json
  {"job_id": "...", "accepted": ["handbook.pdf"], "skipped": [], "rag_slug": "hr-docs", "corr_id": "..."}
  ```
- **Follow-up:** Poll `/api/jobs/{job_id}` until status is `done`. When complete, `/api/rag/docs` reflects the new documents.

### `GET /api/rag/docs?rag_slug=<slug>`
- **Access:** Admin.
- **Response:** Metadata for every stored document, including status (`uploaded`, `chunking`, `chunked`, `indexing`, `indexed`, `error`), `size_bytes`, `chunk_count`, and timestamps.

### `GET /api/rag/index/status?rag_slug=<slug>`
- **Access:** Admin.
- **Response:** Index health summary (`status`, `progress`, totals, `error` string if set). Useful for progress bars while jobs run.

### `POST /api/rag/rebuild`
- **Access:** Admin.
- **Body:** `{"rag_slug": "hr-docs"}`
- **Response 202:** `{"job_id": "...", "corr_id": "..."}`. Reprocesses all non-error docs and rebuilds the FAISS index.

### `POST /api/rag/reset`
- **Access:** Admin.
- **Body:** `{"rag_slug": "hr-docs", "confirm": true}` (confirmation is mandatory).
- **Response 202:** Job ID. The job removes all uploaded files, vector payloads, and doc metadata, leaving the RAG shell intact.
- **Follow-up:** Use `/api/admin/rag/delete` to completely remove the RAG if needed.

### `POST /api/admin/rag/delete`
- **Access:** Admin (`Authorization: Bearer <jwt>` required).
- **Body:**
  ```json
  {
    "rag_slug": "hr-docs",
    "confirmation": "hr-docs"
  }
  ```
- **Response 200:**
  ```json
  {
    "deleted": true,
    "rag_slug": "hr-docs",
    "corr_id": "..."
  }
  ```
- **Validations & errors:** `422` when `confirmation` does not exactly match `rag_slug`, `404` if the slug is unknown, `401/403` for auth failures, `409` reserved for future lock handling, and `500` for unexpected cleanup issues (all error bodies use the standard envelope with `error.code`, `error.message`, and `corr_id`).
- **Side effects:** Synchronously removes the Mongo RAG record, document metadata, FAISS index files under `INDEX_DIR/<slug>`, uploads under `UPLOAD_DIR/<slug>`, and strips the slug from each user’s `rags` array. After success the slug vanishes from `/api/rag/list`, `/api/admin/rag/list`, `/api/rag/docs`, `/api/rag/users`, etc. No follow-up job polling is needed.

## Chat
### `POST /api/chat`
- **Access:** Authenticated users with access to `rag_slug`.
- **Body:**
  ```json
  {
    "rag_slug": "hr-docs",
    "messages": [
      {"role": "user", "content": "What is our vacation policy?"}
    ],
    "opts": {"top_k": 6, "temperature": 0.2, "max_tokens": 500}
  }
  ```
- **Response 200:** 
  ```json
  {
    "answer": "Question: ...",
    "citations": [{"doc_id": "...", "filename": "handbook.pdf", "pages": [1], "snippet": "..."}],
    "usage": {"tokens_in": 42, "tokens_out": 180},
    "corr_id": "..."
  }
  ```
- **Failures:** If the user lacks access to the slug, receive `403` with `FORBIDDEN`.
- **Formatting:** `answer` is GitHub-flavored Markdown; render it client-side (tables, headings, bold, etc. may appear). Tables are normalized to GitHub format (blank line before the block, header/separator/data rows on separate lines). Inline sources are omitted—reuse the `citations` event for attribution.
- **Notes:** When `OPENAI_API_KEY` is configured, ChatFleet calls the configured `CHAT_MODEL` via the background job runner to synthesise the answer from retrieved snippets. Without credentials, the service falls back to a deterministic summary of the most relevant document.

### `POST /api/chat/stream`
- **Access:** Same as `/api/chat`.
- **Response:** `text/event-stream`. Event order:
  1. `event: ready` with `{ "corr_id": "..." }`
  2. Multiple `event: chunk` with `{ "delta": "<text>", "corr_id": "..." }`
  3. `event: citations` with `{ "citations": [...], "corr_id": "..." }`
  4. `event: done` with `{ "usage": { "tokens_in": 42, "tokens_out": 180 }, "corr_id": "..." }`
  5. `event: ping` heartbeat (payload `{}`)
- Frontend must treat payload lines as JSON and stop reading after `done`.
- Each `chunk.delta` string contains raw Markdown fragments (including trailing newlines); append them in order before parsing/rendering on the frontend. The backend never inlines sources—use the `citations` payload to add a dedicated “Sources” section client-side.
- Internally the backend waits for the background LLM job to complete, then streams the composed answer in 240‑character chunks so the UI can display incremental progress.

### `POST /api/public/chat`
- **Access:** Public (no auth) but only for RAGs with `visibility="public"`.
- **Body/Response:** Same contract as `/api/chat`.

### `POST /api/public/chat/stream`
- **Access:** Public (no auth) for public RAGs.
- **Behavior:** Same event contract as `/api/chat/stream`.

## Jobs
### `GET /api/jobs/{job_id}`
- **Access:** Any authenticated user (no admin gate).
- **Response 200:** 
  ```json
  {
    "job_id": "...",
    "type": "RAG_INDEX" | "RAG_REBUILD" | "RAG_RESET",
    "status": "queued" | "running" | "done" | "error",
    "progress": 0.0,
    "started_at": "...",
    "finished_at": "...",
    "result": {"total_chunks": 123, "dimension": 1536},
    "error": null,
    "corr_id": "..."
  }
  ```
- **Errors:** Unknown job returns `404` with `JOB_NOT_FOUND`.

## Admin Config
### `GET /api/admin/config`
- **Access:** Admin.
- **Response:** Current model settings and server directories.
  ```json
  {
    "chat_model": "gpt-4o-mini",
    "embed_model": "text-embedding-3-small",
    "index_dir": "/var/lib/chatfleet/faiss",
    "upload_dir": "/var/lib/chatfleet/uploads",
    "max_upload_mb": 50,
    "corr_id": "..."
  }
  ```

## Frontend Workflows
1. **User onboarding:** Call `/api/auth/register` or have an admin create the user, then assign RAG access via `/api/rag/users/add`.
2. **Creating a RAG:** Call `POST /api/rag`, assign users, upload PDFs/DOCX/ODT/TXT with `/api/rag/upload`, watch the job via `/api/jobs/{job_id}`, and display `GET /api/rag/index/status` for progress.
3. **Updating RAG content:** For new docs, reuse `/api/rag/upload`. For reindexing after bulk changes, trigger `/api/rag/rebuild`.
4. **Resetting or deleting:** Use `/api/rag/reset` to purge documents; follow up with manual Mongo deletion if the slug should disappear entirely.
5. **Chat UI:** Determine available slugs via `/api/rag/list`, start streaming chats with `/api/chat/stream`, and surface citations/corr_id in telemetry logs.

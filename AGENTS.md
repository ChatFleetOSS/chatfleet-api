// generated-by: codex-agent 2025-02-17T12:00:00Z
# Repository Guidelines

This backend powers ChatFleet’s multi-RAG services with FastAPI and contract tooling. Use these notes to keep contributions consistent and reviewable.

## Project Structure & Module Organization
- `main.py` boots the FastAPI app and wires middleware, startup hooks, and graceful shutdown logic.
- `app/` hosts the Python domain code: `core/` (config, logging, corr IDs), `routes/` (HTTP routers), `services/` (business flows), `models/` (Pydantic DTOs), and `utils/` (helpers).
- `schemas/` contains the canonical Zod definitions; import them into providers, tests, and generated clients rather than copying shapes.
- `tests/contract/` delivers Pact suites; `pacts/` and `pact-logs/` hold generated artifacts for review.

## Build, Test, and Development Commands
- `uvicorn main:app --reload` runs the FastAPI server with hot reload for local development.
- `npm test` executes the Jest suite.
- `npm run pact:consumer` exercises consumer contracts under `tests/contract/consumer`.
- `npm run pact:provider` verifies the provider implementation via `ts-node`.
- `npm run pact:all` chains both contract stages for pre-merge validation.

## Coding Style & Naming Conventions
- Python code is formatted with `black` (line length 88) and should stay fully type-hinted.
- TypeScript files use ES2022 modules, two-space indentation, and semicolons; avoid `any` by refining Zod schema inference.
- Name files in kebab-case (`chat-history-service.ts`), classes in PascalCase, constants in SCREAMING_SNAKE_CASE.
- Place `// generated-by: codex-agent <timestamp>` at the top of every generated artifact to preserve traceability.

## Testing Guidelines
- Keep consumer and provider pact specs alongside implementation changes; mirror contract names with the operation under test (e.g., `chat-history.spec.ts`).
- Extend the Jest suite whenever schemas or response envelopes change, and regenerate Pact files before committing.
- Maintain 95 %+ non-`any` coverage by deriving types from `schemas/index.ts`; fail fast on `safeParse` errors.

## Commit & Pull Request Guidelines
- Use focused commits with imperative summaries (`feat: add chat history contract`) and include body details when altering schemas or auth.
- Reference tracking issues or task IDs in the footer, and note any follow-up work.
- Before opening a PR, run the full pact chain, attach log snippets for failures, and list manual QA steps or screenshots when UI clients are impacted.

## Security & Configuration
- Never source real secrets; rely on `.env.example` and ensure new variables are validated through the existing Zod config gate.
- Background jobs must defer external calls (OpenAI, S3, etc.) to queued workers; document any new queue topics in the PR.

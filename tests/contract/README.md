// generated-by: codex-agent 2025-02-15T00:45:00Z
# Contract Testing Guide

## Prerequisites
- Node.js 18+
- `npm install` at the repository root (installs pact + jest dependencies)
- FastAPI backend running locally on `http://localhost:8000` for provider verification

## Commands
- `npm run pact:consumer` — generates/updates pact files in `./pacts/`
- `npm run pact:provider` — verifies the generated pact(s) against the live backend
- `npm run pact:all` — convenience wrapper that runs consumer then provider checks

## Notes
- Pact files are emitted into the `pacts/` directory (ignored by git by default). Share them with the provider build or CI.
- Set `PROVIDER_BASE_URL` when verifying against a non-local backend.
- Use `PROVIDER_VERSION` to stamp verification results when publishing to a broker.

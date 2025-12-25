#!/usr/bin/env bash
# Simple dev runner for the ChatFleet backend.
# - Loads repo env (JWT, etc.)
# - Forces localhost Mongo for host dev
# - Ensures a venv and deps
# - Starts uvicorn with --reload

set -Eeuo pipefail
IFS=$'\n\t'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

note() { printf "[dev] %s\n" "$*"; }
warn() { printf "[dev] WARN: %s\n" "$*" >&2; }
die()  { printf "[dev] ERROR: %s\n" "$*" >&2; exit 1; }

# 1) Load repo-root env (JWT, models, etc.) if present
if [[ -f "$REPO_ROOT/.env" ]]; then
  note "Loading $REPO_ROOT/.env"
  set -a; source "$REPO_ROOT/.env"; set +a
fi

# 2) Load backend-specific env if present (lowest priority)
if [[ -f "$SCRIPT_DIR/.env" ]]; then
  note "Loading $SCRIPT_DIR/.env"
  set -a; source "$SCRIPT_DIR/.env"; set +a
fi

# 3) Prefer localhost Mongo for host dev if not explicitly set otherwise
if [[ -z "${MONGO_URI:-}" || "$MONGO_URI" == mongodb://mongo:* || "$MONGO_URI" == mongodb://mongo* ]]; then
  export MONGO_URI="mongodb://localhost:27017/chatfleet"
fi
export AUTO_START_MONGO="${AUTO_START_MONGO:-0}"
export CORS_ORIGINS="${CORS_ORIGINS:-http://localhost:3000}"

# 4) Ensure strong JWT secret (backend enforces >=32 chars)
if [[ -z "${JWT_SECRET:-}" || ${#JWT_SECRET} -lt 32 ]]; then
  if command -v openssl >/dev/null 2>&1; then
    export JWT_SECRET="$(openssl rand -base64 48 | tr -d '\n')"
    note "Generated JWT_SECRET for dev session"
  else
    die "JWT_SECRET missing/weak and openssl not available. Set JWT_SECRET in $REPO_ROOT/.env (>=32 chars)."
  fi
fi

note "Using MONGO_URI=$MONGO_URI"
note "AUTO_START_MONGO=$AUTO_START_MONGO"
note "CORS_ORIGINS=$CORS_ORIGINS"

# 5) Check Mongo reachability (best-effort hint)
host_port="$(python3 - <<'PY'
import os, urllib.parse as u
uri=os.environ.get('MONGO_URI','mongodb://localhost:27017/chatfleet')
p=u.urlparse(uri)
host=p.hostname or 'localhost'
port=p.port or 27017
print(f"{host}:{port}")
PY
)"
if command -v nc >/dev/null 2>&1; then
  host="${host_port%:*}"; port="${host_port#*:}"
  if ! nc -z "$host" "$port" >/dev/null 2>&1; then
    warn "MongoDB not reachable at $host:$port. Start it (e.g.,: docker start cf-mongo or docker run -d --name cf-mongo -p 27017:27017 mongo:6)"
  fi
fi

# 6) Ensure venv + deps
if [[ -d "$SCRIPT_DIR/.venv" ]]; then
  # shellcheck disable=SC1091
  source "$SCRIPT_DIR/.venv/bin/activate"
else
  command -v python3 >/dev/null 2>&1 || die "python3 not found. Install Python 3.11+."
  note "Creating virtualenv in $SCRIPT_DIR/.venv"
  python3 -m venv "$SCRIPT_DIR/.venv"
  # shellcheck disable=SC1091
  source "$SCRIPT_DIR/.venv/bin/activate"
  pip install --upgrade pip >/dev/null
  pip install -r "$SCRIPT_DIR/requirements.txt"
fi

if ! command -v uvicorn >/dev/null 2>&1; then
  pip install uvicorn >/dev/null
fi

PORT="${PORT:-8000}"
note "Starting uvicorn on :$PORT (reload)"
exec uvicorn main:app --reload --port "$PORT"


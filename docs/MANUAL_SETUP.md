# ChatFleet — Manual Backend + Frontend Setup (No Docker)

Run the API, Mongo, and Web UI locally without Compose or the installer.

## Prerequisites
- Python 3.11+ and `pip`
- Node.js 20+ and `npm`
- MongoDB 7+ running on `mongodb://localhost:27017`
- Optional: Local OpenAI-compatible LLM (e.g., llama-server/vLLM) at `http://127.0.0.1:2242/v1`

## Clone Repos
```bash
git clone https://github.com/ChatFleetOSS/chatfleet-api.git
git clone https://github.com/ChatFleetOSS/chatfleet-web.git
```

## Backend (API)
```bash
cd chatfleet-api
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Set env vars (example: local LLM; adjust paths/models as needed):
```bash
export MONGO_URI="mongodb://localhost:27017/chatfleet"
export JWT_SECRET="$(python - <<'PY' import secrets;print(secrets.token_hex(32)) PY)"
export CHAT_MODEL="unsloth/gpt-oss-20b-GGUF:Q8_K_XL"   # or your chat model id
export EMBED_MODEL="BAAI/bge-m3"
export INDEX_DIR="$PWD/var/faiss"
export UPLOAD_DIR="$PWD/var/uploads"
# If using a local LLM endpoint:
export BASE_URL="http://127.0.0.1:2242/v1"
# If using OpenAI instead, unset BASE_URL and set OPENAI_API_KEY.
```

Seed an admin (email `admin@chatfleet.local`, password `adminpass`):
```bash
MONGO_URI="$MONGO_URI" python scripts/seed_admin.py
```

Run the API:
```bash
uvicorn main:app --reload --port 8000
```

## Frontend (Web)
```bash
cd ../chatfleet-web
npm ci
```

Create `.env.local`:
```
NEXT_PUBLIC_API_BASE=/backend-api
```
(Dev server proxies `/backend-api` to `http://localhost:8000/api` per `next.config.ts`.)

Run the frontend:
```bash
npm run dev   # http://localhost:3000
```

## Configure LLM/Embeddings in the UI
1) Open http://localhost:3000 and log in with `admin@chatfleet.local` / `adminpass`.
2) Admin → Settings:
   - Local LLM: Provider `vllm`, Base URL `http://127.0.0.1:2242/v1`, Chat model `unsloth/gpt-oss-20b-GGUF:Q8_K_XL`, Embedding provider `local`, Embedding model `BAAI/bge-m3`.
   - OpenAI: Provider `openai`, set API key, chat model, and embed model accordingly.
3) Save and run “Test chat” and “Test embed”.

## Notes
- Mongo must be running locally (`mongod --dbpath <path> --bind_ip localhost`) or via your OS service.
- INDEX_DIR and UPLOAD_DIR must be writable; defaults above keep data under `chatfleet-api/var`.
- For a fully local stack (no OpenAI), ensure no OPENAI_API_KEY is set and that the LLM endpoint is reachable on `BASE_URL`.

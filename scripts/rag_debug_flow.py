from __future__ import annotations

import asyncio
import os
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import faiss
import numpy as np
from PyPDF2 import PdfReader

from app.services.chunker import chunk_text
from app.services.chat import _format_hits_for_prompt
from app.services.vectorstore import ChunkRecord
from app.services.embeddings import embed_texts
from scripts.test_vllm import vllm_stream_chat, get_model_name, BASE_URL as TEST_VLLM_BASE_URL
from app.services.runtime_config import get_llm_config
from pymongo.errors import ServerSelectionTimeoutError


os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", TEST_VLLM_BASE_URL)
VLLM_API_KEY = os.getenv("VLLM_API_KEY", "qjlhqdjlshilejnqe1131245dnjqdhfled")
CHAT_MODEL = os.getenv("CHAT_MODEL")
QUERY = os.getenv("QUERY", "qu'est ce que le syntec?")
TOP_K = int(os.getenv("TOP_K", "3"))
DOCS_DIR = os.getenv("DOCS_DIR")
MAX_CONTEXT = int(os.getenv("MAX_CONTEXT", "2000"))
VLLM_TIMEOUT = float(os.getenv("VLLM_TIMEOUT", "120"))
STREAM_CHAT = os.getenv("STREAM_CHAT", "1") == "1"
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "256"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/chatfleet")


@dataclass(frozen=True)
class Doc:
    doc_id: str
    filename: str
    text: str


def log(step: str, message: str) -> None:
    print(f"[{step}] {message}")


def load_docs_from_dir(path: str) -> List[Doc]:
    root = Path(path)
    if not root.exists():
        raise RuntimeError(f"DOCS_DIR not found: {root}")
    docs: List[Doc] = []
    for file_path in sorted(root.rglob("*.txt")):
        content = file_path.read_text(encoding="utf-8").strip()
        if content:
            rel = str(file_path.relative_to(root))
            docs.append(Doc(doc_id=rel, filename=file_path.name, text=content))
    for file_path in sorted(root.rglob("*.pdf")):
        content = extract_pdf_text(str(file_path)).strip()
        if content:
            rel = str(file_path.relative_to(root))
            docs.append(Doc(doc_id=rel, filename=file_path.name, text=content))
    return docs


def default_docs() -> List[Doc]:
    return [
        Doc(
            doc_id="syntec",
            filename="syntec",
            text=(
                "Syntec est une convention collective française qui encadre "
                "les conditions de travail dans les bureaux d'études, "
                "les cabinets de conseil et les services informatiques."
            ),
        ),
        Doc(
            doc_id="vacances",
            filename="vacances",
            text=(
                "La politique de congés précise le calcul des jours acquis "
                "et la procédure de validation par le manager."
            ),
        ),
        Doc(
            doc_id="securite",
            filename="securite",
            text=(
                "La sécurité des informations impose l'usage du MFA et le "
                "chiffrement des documents sensibles."
            ),
        ),
    ]


def extract_pdf_text(path: str) -> str:
    reader = PdfReader(path)
    contents = []
    for page in reader.pages:
        try:
            snippet = page.extract_text() or ""
        except Exception:
            snippet = ""
        contents.append(snippet)
    return "\n".join(contents)


def build_context(hits: List[Tuple[ChunkRecord, float]], max_chars: int) -> str:
    blocks = []
    total = 0
    for record, score in hits:
        chunk = f"[{record.doc_id} | score={score:.3f}]\n{record.text}\n"
        if total + len(chunk) > max_chars:
            if not blocks:
                blocks.append(chunk[:max_chars])
                total = len(blocks[0])
            break
        blocks.append(chunk)
        total += len(chunk)
    return "\n---\n".join(blocks)


async def main() -> None:
    log("config", f"VLLM_BASE_URL={VLLM_BASE_URL}")
    log("config", f"CHAT_MODEL={CHAT_MODEL or '(auto)'}")
    log("config", f"TOP_K={TOP_K}")
    log("query", QUERY)
    if not VLLM_API_KEY:
        log("config", "VLLM_API_KEY not set; vLLM may reject auth.")

    log("config", f"MONGO_URI={MONGO_URI}")
    try:
        cfg = await get_llm_config()
    except ServerSelectionTimeoutError as exc:
        raise RuntimeError(
            "MongoDB is not reachable. Set MONGO_URI to a reachable instance "
            "(e.g. mongodb://localhost:27017/chatfleet) and retry."
        ) from exc
    log("config", f"LLM provider={cfg.provider} embed_provider={getattr(cfg, 'embed_provider', 'openai')}")
    log("config", f"embed_model={cfg.embed_model}")
    if getattr(cfg, "embed_provider", "openai") != "local":
        raise RuntimeError("embed_provider is not local. Run verify_backend_flow.py with EMBED_PROVIDER=local.")

    if DOCS_DIR:
        log("docs", f"Loading docs from {DOCS_DIR}")
        docs = load_docs_from_dir(DOCS_DIR)
    else:
        log("docs", "Using default embedded docs")
        docs = default_docs()

    if not docs:
        raise RuntimeError("No documents loaded.")

    chunk_records: List[ChunkRecord] = []
    all_chunks: List[str] = []
    for doc in docs:
        chunks = chunk_text(doc.text)
        log("chunk", f"{doc.filename}: {len(chunks)} chunks")
        for idx, chunk in enumerate(chunks):
            chunk_records.append(
                ChunkRecord(
                    doc_id=doc.doc_id,
                    filename=doc.filename,
                    chunk_index=idx,
                    text=chunk,
                )
            )
            all_chunks.append(chunk)

    if not all_chunks:
        raise RuntimeError("No textual chunks extracted.")

    log("embed", f"Embedding {len(all_chunks)} chunks via chatfleet embeddings")
    embeddings = await embed_texts(all_chunks)
    matrix = np.asarray(embeddings, dtype="float32")
    log("embed", f"Chunk vectors shape: {matrix.shape}")
    faiss.normalize_L2(matrix)
    index = faiss.IndexFlatIP(matrix.shape[1])
    index.add(matrix)
    log("index", "FAISS index built")

    log("embed", "Embedding query")
    q_vecs = await embed_texts([QUERY])
    q_vec = np.asarray(q_vecs[0], dtype="float32")[None, :]
    faiss.normalize_L2(q_vec)
    log("embed", f"Query vector shape: {q_vec.shape}")

    scores, indices = index.search(q_vec, TOP_K)
    hits: List[Tuple[ChunkRecord, float]] = []
    for idx, score in zip(indices[0].tolist(), scores[0].tolist()):
        if idx < 0 or idx >= len(chunk_records):
            continue
        hits.append((chunk_records[idx], float(score)))

    log("retrieve", f"Top {len(hits)} hits")
    for record, score in hits:
        snippet = textwrap.shorten(record.text, width=120, placeholder="…")
        log("retrieve", f"{record.filename}#{record.chunk_index} score={score:.3f} text={snippet}")

    context = build_context(hits, MAX_CONTEXT)
    log("context", f"Context chars={len(context)} (max={MAX_CONTEXT})")
    log("context", _format_hits_for_prompt([(score, record) for record, score in hits]))

    model_id = CHAT_MODEL or get_model_name()
    log("llm", f"Using model: {model_id}")

    formatted_hits = _format_hits_for_prompt([(score, record) for record, score in hits])
    messages = [
        {
            "role": "system",
            "content": (
                "You are ChatFleet, an assistant that answers with factual precision. "
                "Rely only on the provided knowledge snippets. If the snippets do not "
                "contain the answer, say you are unsure."
            ),
        },
        {
            "role": "system",
            "content": (
                "Knowledge snippets (do not quote outside of these sources):\n"
                f"{formatted_hits}"
            ),
        },
        {
            "role": "system",
            "content": (
                "Respond in GitHub-flavored Markdown. Use short sections, bullet lists, and tables "
                "to present data clearly. When emitting tables, insert a single blank line before the block, keep rows contiguous "
                "with one newline per row (header, separator, and body rows), and keep cells pipe-separated. "
                "Use a single blank line between paragraphs, and avoid extra blank lines inside tables or lists. "
                "Do not include a dedicated 'Sources' section; supporting material is sent separately."
            ),
        },
        {"role": "user", "content": QUERY},
    ]
    log("llm", f"Sending chat completion request (timeout={VLLM_TIMEOUT}s, stream={STREAM_CHAT})")
    if STREAM_CHAT:
        answer = vllm_stream_chat(messages, model_id).strip()
    else:
        raise RuntimeError("STREAM_CHAT=0 is disabled here; use test_vllm.py for non-streaming calls.")


if __name__ == "__main__":
    asyncio.run(main())

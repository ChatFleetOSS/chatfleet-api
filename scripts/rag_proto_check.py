from __future__ import annotations

import json
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from openai import OpenAI


# ------------ Config ------------
VLLM_BASE_URL = "http://127.0.0.1:2242/v1"
VLLM_API_KEY = "qjlhqdjlshilejnqe1131245dnjqdhfled"
CHAT_MODEL = None  # set to a model id string to avoid /v1/models call
What is 

# ------------ Data structures ------------
@dataclass(frozen=True)
class Doc:
    id: str
    text: str
    meta: dict


# ------------ Embeddings ------------
class Embedder:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: List[str]) -> np.ndarray:
        if not texts:
            raise ValueError("embed(): texts is empty")
        vecs = self.model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
        vecs = vecs.astype("float32")
        if vecs.ndim != 2:
            raise RuntimeError(f"Unexpected embeddings shape: {vecs.shape}")
        return vecs


# ------------ Vector store (FAISS) ------------
class FaissStore:
    def __init__(self, dim: int) -> None:
        # cosine similarity via inner product because we normalize embeddings
        self.index = faiss.IndexFlatIP(dim)
        self.docs: List[Doc] = []

    def add(self, doc_vecs: np.ndarray, docs: List[Doc]) -> None:
        if doc_vecs.shape[0] != len(docs):
            raise ValueError("add(): vectors count != docs count")
        if doc_vecs.dtype != np.float32:
            raise ValueError("add(): vectors must be float32")
        self.index.add(doc_vecs)
        self.docs.extend(docs)

    def search(self, q_vec: np.ndarray, k: int = 5) -> List[Tuple[Doc, float]]:
        if q_vec.shape[0] != 1:
            raise ValueError("search(): q_vec must have shape (1, dim)")
        D, I = self.index.search(q_vec, k)
        hits: List[Tuple[Doc, float]] = []
        for idx, score in zip(I[0].tolist(), D[0].tolist()):
            if idx < 0:
                continue
            hits.append((self.docs[idx], float(score)))
        return hits

    def save(self, index_path: str, docs_path: str) -> None:
        faiss.write_index(self.index, index_path)
        payload = [{"id": d.id, "text": d.text, "meta": d.meta} for d in self.docs]
        with open(docs_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    @staticmethod
    def load(index_path: str, docs_path: str) -> "FaissStore":
        index = faiss.read_index(index_path)
        with open(docs_path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        if not payload:
            raise RuntimeError("load(): docs store is empty")

        dim = index.d
        store = FaissStore(dim)
        store.index = index
        store.docs = [Doc(id=p["id"], text=p["text"], meta=p.get("meta", {})) for p in payload]
        return store


# ------------ RAG prompt building ------------
def build_context(hits: List[Tuple[Doc, float]], max_chars: int = 6000) -> str:
    blocks = []
    total = 0
    for doc, score in hits:
        chunk = f"[{doc.id} | score={score:.3f}]\n{doc.text}\n"
        if total + len(chunk) > max_chars:
            break
        blocks.append(chunk)
        total += len(chunk)
    return "\n---\n".join(blocks)


# ------------ vLLM chat ------------
def get_chat_model_id(client: OpenAI) -> str:
    if CHAT_MODEL:
        return CHAT_MODEL
    models = client.models.list().data
    if not models:
        raise RuntimeError("No models returned by /v1/models")
    return models[0].id


def chat_with_rag(client: OpenAI, model: str, user_query: str, context: str) -> str:
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Use the provided context when relevant."},
        {"role": "system", "content": f"CONTEXT:\n{context}" if context else "CONTEXT: (none)"},
        {"role": "user", "content": user_query},
    ]
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        stream=False,
        extra_body={"repetition_penalty": 1.1},
    )
    return resp.choices[0].message.content or ""


# ------------ Demo ------------
if __name__ == "__main__":
    # 1) Build a tiny corpus (replace with your scraped items later)
    docs = [
        Doc(id="d1", text="Crowdaa is a white-label community app platform.", meta={"source": "note"}),
        Doc(id="d2", text="PurePuls curates news and helps founders post relevant insights.", meta={"source": "note"}),
        Doc(id="d3", text="Kodaii generates FastAPI backends from prompts with tests and infra.", meta={"source": "note"}),
    ]

    embedder = Embedder()
    doc_vecs = embedder.embed([d.text for d in docs])

    store = FaissStore(dim=doc_vecs.shape[1])
    store.add(doc_vecs, docs)

    # 2) Query → retrieve
    q = "What does Kodaii do?"
    q_vec = embedder.embed([q])
    hits = store.search(q_vec, k=3)
    context = build_context(hits)

    # 3) Ask vLLM using retrieved context
    client = OpenAI(base_url=VLLM_BASE_URL, api_key=VLLM_API_KEY)
    model_id = get_chat_model_id(client)
    answer = chat_with_rag(client, model_id, q, context)

    print("\n--- Context ---\n", context)
    print("\n--- Answer ---\n", answer)

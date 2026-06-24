# generated-by: codex-agent 2026-05-05T00:00:00Z
from __future__ import annotations

import json
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from app.models.chat import ChatMessage, ChatRequest

with patch.dict(sys.modules, {"sentence_transformers": None}):
    from app.services import chat
from app.services import lexical_search, vectorstore
from app.services.hybrid_retrieval import hybrid_retrieve
from app.services.vectorstore import build_index, persist_doc_payload, query_index


RAG_SLUG = "hybrid-smoke-rh"
DOC_ID = "11111111-1111-1111-1111-111111111111"
QUERY_SEMANTIC_LEAVE = [1.0, 0.0, 0.0, 0.0, 0.0]

SMOKE_CHUNKS = [
    {
        "text": (
            "Congé parental: les salariés ayant douze mois d'ancienneté peuvent "
            "demander seize semaines de congé, avec préavis écrit de trente jours."
        ),
        "page_start": 1,
        "page_end": 1,
    },
    {
        "text": (
            "Procédure disciplinaire R-2024-17: toute convocation doit mentionner "
            "le motif précis et laisser cinq jours ouvrables avant l'entretien."
        ),
        "page_start": 2,
        "page_end": 2,
    },
    {
        "text": (
            "Clause de confidentialité: Ada Martin valide les accords NDA avant "
            "signature par les prestataires juridiques."
        ),
        "page_start": 3,
        "page_end": 3,
    },
    {
        "text": (
            "Paie 2026: les primes exceptionnelles sont versées le 2026-05-05 "
            "sous le code PAY-889."
        ),
        "page_start": 4,
        "page_end": 4,
    },
    {
        "text": (
            "Résumé d'évaluation: l'élève prioritaire bénéficie d'un suivi RH "
            "mensuel et d'un référent dédié."
        ),
        "page_start": 5,
        "page_end": 5,
    },
    {
        "text": (
            "Budget formation: les demandes inférieures à 1200 euros sont "
            "approuvées par le manager direct."
        ),
        "page_start": 6,
        "page_end": 6,
    },
    {
        "text": (
            "Astreinte informatique: le ticket OPS-77 impose une réponse sous "
            "quinze minutes les jours ouvrés."
        ),
        "page_start": 7,
        "page_end": 7,
    },
    {
        "text": (
            "Télétravail: deux jours par semaine sont autorisés après validation "
            "écrite du responsable d'équipe."
        ),
        "page_start": 8,
        "page_end": 8,
    },
]

SMOKE_EMBEDDINGS = [
    [1.0, 0.0, 0.0, 0.0, 0.0],
    [0.35, 0.94, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 1.0, 0.0],
    [0.92, 0.39, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 1.0],
    [0.7, 0.0, 0.7, 0.0, 0.0],
]


async def _fake_write_system_log(*_args, **_kwargs) -> None:
    return None


async def _fake_generate_chat_completion(messages, temperature, max_tokens):
    context = messages[-1]["content"]
    if "R-2024-17" in context:
        return (
            "La procédure R-2024-17 impose de mentionner le motif précis "
            "et de laisser cinq jours ouvrables avant l'entretien.",
            24,
        )
    return "Je n'ai pas cette information dans les extraits fournis.", 12


async def _fake_embed_text(text: str) -> list[float]:
    if "R-2024-17" in text:
        return QUERY_SEMANTIC_LEAVE
    if "congé" in text.casefold() or "parental" in text.casefold():
        return QUERY_SEMANTIC_LEAVE
    return [0.0, 1.0, 0.0, 0.0, 0.0]


async def _fake_retrieve_hits(rag_slug, question, top_k, retrieval_config=None):
    vector = await _fake_embed_text(question)
    return hybrid_retrieve(
        rag_slug,
        vector,
        question,
        top_k,
        min_semantic_score=0.2,
        semantic_search=query_index,
        retrieval_config=retrieval_config,
    )


class HybridRetrievalSmokeTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.index_dir = Path(self.tmp.name) / "indexes"
        self.upload_dir = Path(self.tmp.name) / "uploads"
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.runtime_patch = patch.object(
            vectorstore,
            "get_runtime_overrides_sync",
            return_value=(self.index_dir, self.upload_dir, 50, 0.2, 3),
        )
        self.runtime_patch.start()
        lexical_search._CACHE.clear()
        persist_doc_payload(
            RAG_SLUG,
            DOC_ID,
            "rh-juridique.pdf",
            SMOKE_CHUNKS,
            SMOKE_EMBEDDINGS,
        )
        chunks, dimension = build_index(RAG_SLUG)
        self.assertEqual(chunks, 8)
        self.assertEqual(dimension, 5)

    def tearDown(self) -> None:
        lexical_search._CACHE.clear()
        self.runtime_patch.stop()
        self.tmp.cleanup()

    def test_smoke_semantic_paraphrase_and_exact_reference_gain(self) -> None:
        semantic_leave = query_index(
            RAG_SLUG,
            QUERY_SEMANTIC_LEAVE,
            top_k=1,
            min_score=0.2,
        )
        self.assertIn("Congé parental", semantic_leave[0][1].text)

        semantic_exact_only = query_index(
            RAG_SLUG,
            QUERY_SEMANTIC_LEAVE,
            top_k=1,
            min_score=0.2,
        )
        self.assertNotIn("R-2024-17", semantic_exact_only[0][1].text)

        hybrid_exact = hybrid_retrieve(
            RAG_SLUG,
            QUERY_SEMANTIC_LEAVE,
            "Que prévoit la référence R-2024-17 ?",
            top_k=1,
            min_semantic_score=0.2,
        )
        self.assertIn("R-2024-17", hybrid_exact.hits[0][1].text)
        self.assertEqual(hybrid_exact.diagnostics.semantic_count, 4)
        self.assertGreaterEqual(hybrid_exact.diagnostics.lexical_count, 1)

        hybrid_mixed = hybrid_retrieve(
            RAG_SLUG,
            QUERY_SEMANTIC_LEAVE,
            "Quels droits pour le congé parental ?",
            top_k=1,
            min_semantic_score=0.2,
        )
        self.assertIn("Congé parental", hybrid_mixed.hits[0][1].text)

    async def test_smoke_chat_and_stream_use_real_index_hybrid_retrieval(self) -> None:
        request = ChatRequest(
            rag_slug=RAG_SLUG,
            messages=[
                ChatMessage(
                    role="user",
                    content="Que prévoit la référence R-2024-17 ?",
                )
            ],
        )
        llm_cfg = SimpleNamespace(
            provider="openai",
            base_url=None,
            temperature_default=0.2,
            top_k_default=3,
        )

        with (
            patch.object(chat, "get_llm_config", return_value=llm_cfg),
            patch.object(chat, "get_api_key", return_value="test-key"),
            patch.object(chat, "get_rag_by_slug", return_value={"system_prompt": ""}),
            patch.object(chat, "_retrieve_hits", side_effect=_fake_retrieve_hits),
            patch.object(
                chat,
                "generate_chat_completion",
                side_effect=_fake_generate_chat_completion,
            ),
            patch.object(chat, "write_system_log", side_effect=_fake_write_system_log),
        ):
            response = await chat.handle_chat(request, user_id="smoke-user")
            self.assertIn("R-2024-17", response.answer)
            self.assertIn("R-2024-17", response.citations[0].snippet)

            events = []
            async for event in chat.stream_chat(request, user_id="smoke-user"):
                events.append(event)

        citation_events = [
            event for event in events if event.startswith("event: citations")
        ]
        self.assertTrue(events[0].startswith("event: ready"))
        self.assertEqual(len(citation_events), 1)
        payload = citation_events[0].split("data: ", 1)[1].strip()
        citations = json.loads(payload)["citations"]
        self.assertIn("R-2024-17", citations[0]["snippet"])
        self.assertTrue(any(event.startswith("event: done") for event in events))

    def test_smoke_hybrid_query_latency_on_representative_local_corpus(self) -> None:
        start = time.perf_counter()
        for _ in range(100):
            result = hybrid_retrieve(
                RAG_SLUG,
                QUERY_SEMANTIC_LEAVE,
                "Que prévoit la référence R-2024-17 ?",
                top_k=3,
                min_semantic_score=0.2,
            )
            self.assertTrue(result.hits)
        elapsed = time.perf_counter() - start
        self.assertLess(elapsed, 1.0)

    def test_faiss_index_cache_reuses_loaded_index_and_invalidates_on_metadata_change(
        self,
    ) -> None:
        with patch.object(
            vectorstore.faiss,
            "read_index",
            wraps=vectorstore.faiss.read_index,
        ) as read_index:
            self.assertTrue(
                query_index(RAG_SLUG, QUERY_SEMANTIC_LEAVE, top_k=1, min_score=0.2)
            )
            self.assertTrue(
                query_index(RAG_SLUG, QUERY_SEMANTIC_LEAVE, top_k=1, min_score=0.2)
            )
            self.assertEqual(read_index.call_count, 1)

            metadata_path = self.index_dir / RAG_SLUG / vectorstore.METADATA_FILE
            entries = json.loads(metadata_path.read_text(encoding="utf-8"))
            entries[0]["text"] = entries[0]["text"] + " Cache invalidation marker."
            metadata_path.write_text(json.dumps(entries), encoding="utf-8")

            self.assertTrue(
                query_index(RAG_SLUG, QUERY_SEMANTIC_LEAVE, top_k=1, min_score=0.2)
            )
            self.assertEqual(read_index.call_count, 2)

    def test_lexical_prewarm_runs_only_when_enabled(self) -> None:
        with (
            patch.object(
                vectorstore,
                "get_retrieval_config_sync",
                return_value=SimpleNamespace(lexical_prewarm=False),
            ),
            patch("app.services.lexical_search.prewarm_lexical_index") as prewarm,
        ):
            build_index(RAG_SLUG)
            prewarm.assert_not_called()

        with (
            patch.object(
                vectorstore,
                "get_retrieval_config_sync",
                return_value=SimpleNamespace(lexical_prewarm=True),
            ),
            patch("app.services.lexical_search.prewarm_lexical_index") as prewarm,
        ):
            build_index(RAG_SLUG)
            prewarm.assert_called_once_with(RAG_SLUG)


if __name__ == "__main__":
    unittest.main()

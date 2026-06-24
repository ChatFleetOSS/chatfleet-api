# generated-by: codex-agent 2026-05-22T00:00:00Z
from __future__ import annotations

import importlib
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from fastapi import HTTPException

with patch.dict(sys.modules, {"sentence_transformers": None}):
    chat = importlib.import_module("app.services.chat")
    hybrid_retrieval = importlib.import_module("app.services.hybrid_retrieval")

from app.models.chat import ChatMessage, ChatRequest
from app.services.vectorstore import ChunkRecord


async def _fake_write_system_log(*_args, **_kwargs) -> None:
    return None


class ChatRetrievalDiagnosticsUnitTest(unittest.IsolatedAsyncioTestCase):
    async def test_no_context_logs_missing_index_diagnostics(self) -> None:
        request = ChatRequest(
            rag_slug="legacy-client",
            messages=[
                ChatMessage(
                    role="user",
                    content="What changed during the upgrade?",
                )
            ],
        )
        retrieval_result = hybrid_retrieval.HybridRetrievalResult(
            hits=[],
            diagnostics=hybrid_retrieval.HybridDiagnostics(
                semantic_count=0,
                lexical_count=0,
                final_count=0,
                semantic_ranks={},
                lexical_ranks={},
                index_missing=True,
                index_error=(
                    "Index not built for rag 'legacy-client' "
                    "(expected index=/var/lib/chatfleet/faiss/legacy-client/index.faiss)"
                ),
            ),
        )
        llm_cfg = SimpleNamespace(
            provider="openai",
            base_url=None,
            temperature_default=0.2,
            top_k_default=3,
            retrieval=None,
        )

        with (
            patch.object(chat, "get_llm_config", return_value=llm_cfg),
            patch.object(chat, "get_api_key", return_value="test-key"),
            patch.object(chat, "get_rag_by_slug", return_value={"system_prompt": ""}),
            patch.object(chat, "_retrieve_hits", return_value=retrieval_result),
            patch.object(chat.retrieval_logger, "warning") as warning,
        ):
            with self.assertRaises(HTTPException) as raised:
                await chat.handle_chat(request, user_id="operator")

        self.assertEqual(raised.exception.status_code, 503)
        self.assertEqual(raised.exception.detail["error"]["code"], "NO_CONTEXT")
        warning.assert_called_once()
        self.assertEqual(warning.call_args.args[0], "chat.retrieval.no_context")
        extra = warning.call_args.kwargs["extra"]
        self.assertTrue(extra["index_missing"])
        self.assertIn("/var/lib/chatfleet/faiss/legacy-client", extra["index_error"])

    async def test_chat_retrieval_log_includes_latency_diagnostics(self) -> None:
        request = ChatRequest(
            rag_slug="support",
            messages=[
                ChatMessage(
                    role="user",
                    content="What is the support policy?",
                )
            ],
        )
        retrieval_result = hybrid_retrieval.HybridRetrievalResult(
            hits=[
                (
                    0.9,
                    ChunkRecord(
                        doc_id="11111111-1111-1111-1111-111111111111",
                        filename="policy.txt",
                        chunk_index=0,
                        text="Support policy requires a response within one day.",
                    ),
                )
            ],
            diagnostics=hybrid_retrieval.HybridDiagnostics(
                semantic_count=4,
                lexical_count=2,
                final_count=1,
                semantic_ranks={("11111111-1111-1111-1111-111111111111", 0): 1},
                lexical_ranks={("11111111-1111-1111-1111-111111111111", 0): 2},
                candidate_k=24,
                embedding_ms=12.5,
                semantic_ms=3.0,
                lexical_ms=4.0,
                fusion_ms=0.5,
                retrieval_ms=20.0,
            ),
        )
        llm_cfg = SimpleNamespace(
            provider="openai",
            base_url=None,
            temperature_default=0.2,
            top_k_default=3,
            retrieval=None,
        )

        with (
            patch.object(chat, "get_llm_config", return_value=llm_cfg),
            patch.object(chat, "get_api_key", return_value="test-key"),
            patch.object(chat, "get_rag_by_slug", return_value={"system_prompt": ""}),
            patch.object(chat, "_retrieve_hits", return_value=retrieval_result),
            patch.object(chat, "_generate_answer_with_job", return_value=("Answer", 7)),
            patch.object(chat, "write_system_log", side_effect=_fake_write_system_log),
            patch.object(chat.retrieval_logger, "info") as retrieval_info,
        ):
            response = await chat.handle_chat(request, user_id="operator")

        self.assertEqual(response.answer, "Answer")
        retrieval_call = next(
            call
            for call in retrieval_info.call_args_list
            if call.args[0] == "chat.retrieval"
        )
        extra = retrieval_call.kwargs["extra"]
        self.assertEqual(extra["candidate_k"], 24)
        self.assertEqual(extra["embedding_ms"], 12.5)
        self.assertEqual(extra["semantic_ms"], 3.0)
        self.assertEqual(extra["lexical_ms"], 4.0)
        self.assertEqual(extra["fusion_ms"], 0.5)
        self.assertEqual(extra["retrieval_ms"], 20.0)


if __name__ == "__main__":
    unittest.main()

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


if __name__ == "__main__":
    unittest.main()

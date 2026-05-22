# generated-by: codex-agent 2026-05-15T00:00:00Z
from __future__ import annotations

import importlib
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

with patch.dict(sys.modules, {"sentence_transformers": None}):
    embeddings = importlib.import_module("app.services.embeddings")


class EmbeddingsFallbackUnitTest(unittest.IsolatedAsyncioTestCase):
    async def test_local_provider_failure_keeps_runtime_embedding_dimension(self) -> None:
        cfg = SimpleNamespace(
            provider="vllm",
            base_url="http://localhost:8001/v1",
            embed_provider="local",
            embed_model=embeddings.LOCAL_EMBED_MODEL_DEFAULT,
        )

        with (
            patch.object(embeddings, "get_llm_config", return_value=cfg),
            patch.object(
                embeddings,
                "_embed_texts_local",
                side_effect=RuntimeError("local model unavailable"),
            ),
            patch.object(embeddings, "_get_embed_client", return_value=None),
            patch.object(embeddings.logger, "exception"),
            patch.object(embeddings.logger, "warning"),
        ):
            vectors = await embeddings.embed_texts(["commune de saint-joseph"])

        self.assertEqual(len(vectors), 1)
        self.assertEqual(len(vectors[0]), embeddings.LOCAL_EMBED_MODEL_DEFAULT_DIM)

    async def test_remote_provider_error_uses_runtime_fallback_dimension(self) -> None:
        cfg = SimpleNamespace(
            provider="openai",
            base_url=None,
            embed_provider="openai",
            embed_model="text-embedding-3-small",
        )
        client = SimpleNamespace(
            embeddings=SimpleNamespace(
                create=lambda **_kwargs: (_ for _ in ()).throw(
                    RuntimeError("provider unavailable")
                )
            )
        )

        with (
            patch.object(embeddings, "get_llm_config", return_value=cfg),
            patch.object(embeddings, "get_api_key", return_value="test-key"),
            patch.object(embeddings, "_get_embed_client", return_value=client),
            patch.object(embeddings.logger, "exception"),
        ):
            vectors = await embeddings.embed_texts(["commune de saint-joseph"])

        self.assertEqual(len(vectors), 1)
        self.assertEqual(len(vectors[0]), embeddings.EMBED_DIM)


if __name__ == "__main__":
    unittest.main()

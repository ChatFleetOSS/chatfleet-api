# generated-by: codex-agent 2026-05-15T00:00:00Z
from __future__ import annotations

import asyncio
import importlib
import sys
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

with patch.dict(sys.modules, {"sentence_transformers": None}):
    embeddings = importlib.import_module("app.services.embeddings")


class EmbeddingsFallbackUnitTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        embeddings._local_models.clear()
        embeddings._local_embed_semaphore = None

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
            patch.object(embeddings, "get_api_key", return_value=None),
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

    async def test_local_encode_respects_configured_concurrency_limit(self) -> None:
        active = 0
        max_active = 0

        class FakeModel:
            def encode(self, texts, normalize_embeddings=True, convert_to_numpy=True):
                nonlocal active, max_active
                active += 1
                max_active = max(max_active, active)
                time.sleep(0.05)
                active -= 1
                return embeddings.np.ones((len(texts), 4), dtype="float32")

        embeddings._local_models["fake"] = FakeModel()
        with (
            patch.dict("os.environ", {"CHATFLEET_EMBED_CONCURRENCY": "1"}),
            patch.object(embeddings, "SentenceTransformer", object),
        ):
            results = await asyncio.gather(
                embeddings._embed_texts_local(["a"], "fake"),
                embeddings._embed_texts_local(["b"], "fake"),
            )

        self.assertEqual(len(results), 2)
        self.assertEqual(max_active, 1)

    async def test_local_model_is_loaded_once_when_requested_concurrently(self) -> None:
        created = 0

        class FakeSentenceTransformer:
            def __init__(self, model_name: str):
                nonlocal created
                created += 1
                time.sleep(0.05)
                self.model_name = model_name

        def load_model() -> object:
            return embeddings._ensure_local_model("fake-concurrent")

        with patch.object(embeddings, "SentenceTransformer", FakeSentenceTransformer):
            loop = asyncio.get_running_loop()
            loaded = await asyncio.gather(
                loop.run_in_executor(None, load_model),
                loop.run_in_executor(None, load_model),
                loop.run_in_executor(None, load_model),
            )

        self.assertEqual(created, 1)
        self.assertIs(loaded[0], loaded[1])
        self.assertIs(loaded[1], loaded[2])


if __name__ == "__main__":
    unittest.main()

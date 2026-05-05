# generated-by: codex-agent 2026-05-05T00:00:00Z
from __future__ import annotations

import unittest
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch

from bson import ObjectId

from app.models.chat import ChatMessage, ChatRequest
from app.models.rag import (
    DEFAULT_RAG_SYSTEM_PROMPT,
    RagCreateRequest,
    RagUpdateRequest,
    normalize_rag_system_prompt,
)
from app.services import chat, rags


class FakeRagCollection:
    def __init__(self) -> None:
        self.docs: dict[str, dict] = {}

    async def insert_one(self, doc: dict) -> SimpleNamespace:
        stored = deepcopy(doc)
        inserted_id = ObjectId()
        stored["_id"] = inserted_id
        self.docs[stored["slug"]] = stored
        return SimpleNamespace(inserted_id=inserted_id)

    async def find_one(self, query: dict) -> dict | None:
        doc = self.docs.get(query["slug"])
        return deepcopy(doc) if doc else None

    async def update_one(self, query: dict, update: dict) -> SimpleNamespace:
        doc = self.docs.get(query["slug"])
        if not doc:
            return SimpleNamespace(matched_count=0)
        doc.update(update.get("$set", {}))
        return SimpleNamespace(matched_count=1)


async def noop_write_system_log(*_args, **_kwargs) -> None:
    return None


class RagSystemPromptUnitTest(unittest.IsolatedAsyncioTestCase):
    def test_empty_prompt_normalizes_to_default(self) -> None:
        self.assertEqual(normalize_rag_system_prompt(None), DEFAULT_RAG_SYSTEM_PROMPT)
        self.assertEqual(normalize_rag_system_prompt("   "), DEFAULT_RAG_SYSTEM_PROMPT)
        self.assertEqual(normalize_rag_system_prompt(" Be concise. "), "Be concise.")

    def test_chat_prompt_uses_custom_rag_prompt_and_keeps_context_rules(self) -> None:
        request = ChatRequest(
            rag_slug="policies",
            messages=[ChatMessage(role="user", content="What is the leave policy?")],
        )
        messages = chat._build_prompt_messages(
            request,
            hits=[],
            system_prompt="Answer in a direct support tone.",
        )

        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[0]["content"], "Answer in a direct support tone.")
        self.assertIn("Utilise UNIQUEMENT le CONTEXTE", messages[1]["content"])

    async def test_create_and_update_rag_prompt_metadata(self) -> None:
        collection = FakeRagCollection()
        creator_id = str(ObjectId())

        with (
            patch.object(rags, "get_collection", return_value=collection),
            patch.object(rags, "write_system_log", noop_write_system_log),
        ):
            created = await rags.create_rag(
                RagCreateRequest(
                    slug="policies",
                    name="Policies",
                    system_prompt="Answer as the HR policy assistant.",
                ),
                creator_id,
            )

            self.assertEqual(created.system_prompt, "Answer as the HR policy assistant.")
            self.assertEqual(
                collection.docs["policies"]["system_prompt"],
                "Answer as the HR policy assistant.",
            )

            updated = await rags.update_rag_metadata(
                RagUpdateRequest(rag_slug="policies", system_prompt=""),
                creator_id,
            )

            self.assertEqual(updated.system_prompt, DEFAULT_RAG_SYSTEM_PROMPT)
            self.assertEqual(
                collection.docs["policies"]["system_prompt"],
                DEFAULT_RAG_SYSTEM_PROMPT,
            )

    def test_update_request_requires_at_least_one_metadata_field(self) -> None:
        with self.assertRaises(ValueError):
            RagUpdateRequest(rag_slug="policies")


if __name__ == "__main__":
    unittest.main()

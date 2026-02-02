import asyncio
import unittest
from unittest.mock import patch

from app.services.rags import (
    _clean_question,
    _dedupe_questions,
    _score_window_text,
    _generate_rag_suggestions,
)
from app.services.vectorstore import ChunkRecord


class SuggestionsUnitTest(unittest.IsolatedAsyncioTestCase):
    def test_clean_question_truncates(self):
        txt = "1)   Ceci est une question très longue " + ("x" * 200)
        cleaned = _clean_question(txt, max_len=50)
        self.assertTrue(len(cleaned) <= 50)
        self.assertFalse(cleaned.startswith("1)"))

    def test_dedupe_questions(self):
        qs = ["Question?", "question ?", "QUESTION ?", "Autre"]
        deduped = _dedupe_questions(qs)
        self.assertEqual(len(deduped), 2)

    def test_score_window_text_policy(self):
        low = _score_window_text("Texte banal sans règles.")
        high = _score_window_text("Il est obligatoire de fournir un justificatif. Étape 1 : faire la demande.")
        self.assertGreater(high, low)

    async def test_generate_rag_suggestions_mocked(self):
        records = [
            ChunkRecord(doc_id="d1", filename="f", chunk_index=i, text=f"Remboursement obligatoire des frais de voyage {i}")
            for i in range(12)
        ]

        def fake_load_index(rag_slug: str):
            return 1536, records

        async def fake_generate_chat_completion(messages, temperature, max_tokens):
            content = "Comment sont remboursés les frais de voyage ?\nQuelle classe est autorisée ?"
            if "Traduis" in messages[0]["content"]:
                content = "How are travel expenses reimbursed?\nWhich class is allowed?"
            return content, 10

        def fake_embed_texts(texts):
            # deterministic simple vectors
            return [[float(i + 1)] * 4 for i, _ in enumerate(texts)]

        with patch("app.services.rags.load_index", fake_load_index), patch(
            "app.services.rags.generate_chat_completion", fake_generate_chat_completion
        ), patch("app.services.rags.embed_texts", side_effect=fake_embed_texts):
            primary, suggestions_en, lang_primary, err, fb = await _generate_rag_suggestions(
                "rag1", "rag1", "desc"
            )
            self.assertIsNone(err)
            self.assertFalse(fb)
            self.assertGreaterEqual(len(primary), 2)
            self.assertEqual(lang_primary, "fr")
            self.assertGreaterEqual(len(suggestions_en), 2)


if __name__ == "__main__":
    unittest.main()

# generated-by: codex-agent 2026-05-05T00:00:00Z
from __future__ import annotations

import unittest
from unittest.mock import patch

from pydantic import ValidationError

from app.models.admin import RetrievalConfig
from app.services import hybrid_retrieval, lexical_search
from app.services.vectorstore import ChunkRecord


def _chunk(
    doc_id: str,
    chunk_index: int,
    text: str,
    filename: str | None = None,
) -> ChunkRecord:
    return ChunkRecord(
        doc_id=doc_id,
        filename=filename or f"{doc_id}.pdf",
        chunk_index=chunk_index,
        text=text,
    )


LEXICAL_FIXTURES = [
    _chunk("common", 0, "General onboarding and common policy information."),
    _chunk("rare", 0, "Escalate incidents containing zyxqv to tier two."),
    _chunk("person", 0, "Ada Lovelace approved the analytical engine memo."),
    _chunk("ref", 0, "Release 2026-05-05 references CF-123 and ISO-27001 controls."),
    _chunk("accent", 0, "Résumé de suivi pour l'ÉLÈVE prioritaire."),
]


class LexicalSearchUnitTest(unittest.TestCase):
    def _search(self, query: str, top_k: int = 3) -> list[tuple[float, ChunkRecord]]:
        index = lexical_search.build_lexical_index(LEXICAL_FIXTURES)
        with patch.object(lexical_search, "load_lexical_index", return_value=index):
            return lexical_search.search_lexical_index("support", query, top_k=top_k)

    def test_tokenize_text_normalizes_accents_case_dates_codes_and_empty_input(
        self,
    ) -> None:
        tokens = lexical_search.tokenize_text(
            "Résumé ÉLÈVE CF-123 ISO-27001 2026-05-05"
        )

        self.assertIn("resume", tokens)
        self.assertIn("eleve", tokens)
        self.assertIn("cf-123", tokens)
        self.assertIn("iso-27001", tokens)
        self.assertIn("2026-05-05", tokens)
        self.assertEqual(lexical_search.tokenize_text("   "), [])

    def test_bm25_exact_rare_term_beats_common_text(self) -> None:
        hits = self._search("zyxqv")

        self.assertGreaterEqual(len(hits), 1)
        self.assertEqual(hits[0][1].doc_id, "rare")

    def test_proper_name_query_matches_name_chunk(self) -> None:
        hits = self._search("Ada Lovelace")

        self.assertGreaterEqual(len(hits), 1)
        self.assertEqual(hits[0][1].doc_id, "person")

    def test_date_code_and_reference_query_matches_reference_chunk(self) -> None:
        hits = self._search("CF-123 ISO-27001 2026-05-05")

        self.assertGreaterEqual(len(hits), 1)
        self.assertEqual(hits[0][1].doc_id, "ref")

    def test_accent_and_case_insensitive_query_matches_accented_chunk(self) -> None:
        hits = self._search("resume eleve")

        self.assertGreaterEqual(len(hits), 1)
        self.assertEqual(hits[0][1].doc_id, "accent")

    def test_empty_query_returns_no_lexical_hits(self) -> None:
        self.assertEqual(self._search("   "), [])


class ReciprocalRankFusionUnitTest(unittest.TestCase):
    def setUp(self) -> None:
        self.a = _chunk("a", 0, "Semantic-only first chunk.")
        self.b = _chunk("b", 0, "Chunk present in both rankings.")
        self.c = _chunk("c", 0, "Lexical-only chunk.")
        self.d = _chunk("d", 0, "Second lexical-only chunk.")

    def test_chunk_present_in_both_lists_ranks_above_single_list_chunks(self) -> None:
        fused = hybrid_retrieval.reciprocal_rank_fusion(
            semantic_hits=[(0.91, self.a), (0.90, self.b)],
            lexical_hits=[(12.0, self.b), (11.0, self.c)],
            top_k=3,
        )

        self.assertEqual([record.doc_id for _, record in fused], ["b", "a", "c"])

    def test_lexical_only_results_are_returned_when_semantic_list_is_empty(
        self,
    ) -> None:
        fused = hybrid_retrieval.reciprocal_rank_fusion(
            semantic_hits=[],
            lexical_hits=[(12.0, self.c), (11.0, self.d)],
            top_k=5,
        )

        self.assertEqual([record.doc_id for _, record in fused], ["c", "d"])

    def test_equal_fusion_scores_keep_stable_first_seen_order(self) -> None:
        fused = hybrid_retrieval.reciprocal_rank_fusion(
            semantic_hits=[(0.91, self.a), (0.90, self.b)],
            lexical_hits=[(12.0, self.b), (11.0, self.a)],
            top_k=2,
        )

        self.assertEqual([record.doc_id for _, record in fused], ["a", "b"])

    def test_duplicate_chunks_are_deduplicated_by_doc_and_chunk(self) -> None:
        fused = hybrid_retrieval.reciprocal_rank_fusion(
            semantic_hits=[(0.91, self.a), (0.90, self.a), (0.80, self.b)],
            lexical_hits=[(12.0, self.a), (11.0, self.c)],
            top_k=5,
        )

        keys = [(record.doc_id, record.chunk_index) for _, record in fused]
        self.assertEqual(len(keys), len(set(keys)))
        self.assertEqual(keys[0], ("a", 0))

    def test_zero_weight_source_does_not_add_its_own_hits(self) -> None:
        fused = hybrid_retrieval.reciprocal_rank_fusion(
            semantic_hits=[(0.91, self.a)],
            lexical_hits=[(12.0, self.c)],
            top_k=5,
            semantic_weight=1.0,
            lexical_weight=0.0,
        )

        self.assertEqual([record.doc_id for _, record in fused], ["a"])


class RetrievalConfigValidationUnitTest(unittest.TestCase):
    def test_rejects_out_of_bounds_and_zero_weights(self) -> None:
        with self.assertRaises(ValidationError):
            RetrievalConfig(rrf_k=0)

        with self.assertRaises(ValidationError):
            RetrievalConfig(semantic_weight=0.0, lexical_weight=0.0)

    def test_accepts_recommended_defaults(self) -> None:
        cfg = RetrievalConfig()

        self.assertEqual(cfg.mode, "hybrid")
        self.assertEqual(cfg.candidate_multiplier, 4)
        self.assertEqual(cfg.candidate_min, 24)


class HybridRetrieveUnitTest(unittest.TestCase):
    def setUp(self) -> None:
        self.semantic = _chunk("semantic", 0, "Semantic-only support chunk.")
        self.both = _chunk(
            "both",
            0,
            "Chunk retrieved by both semantic and lexical search.",
        )
        self.lexical = _chunk("lexical", 0, "Lexical-only support chunk.")

    def test_hybrid_retrieve_fuses_monkeypatched_semantic_and_lexical_hits(
        self,
    ) -> None:
        def fake_query_index(*_args, **_kwargs) -> list[tuple[float, ChunkRecord]]:
            return [(0.91, self.semantic), (0.90, self.both)]

        def fake_search_lexical_index(
            *_args,
            **_kwargs,
        ) -> list[tuple[float, ChunkRecord]]:
            return [(12.0, self.both), (11.0, self.lexical)]

        result = hybrid_retrieval.hybrid_retrieve(
            "support",
            [1.0, 0.0, 0.0],
            "support question",
            top_k=3,
            min_semantic_score=0.2,
            semantic_search=fake_query_index,
            lexical_search=fake_search_lexical_index,
            retrieval_config=RetrievalConfig(candidate_multiplier=2, candidate_min=6),
        )

        self.assertEqual(
            [record.doc_id for _, record in result.hits],
            ["both", "semantic", "lexical"],
        )
        self.assertEqual(result.diagnostics.semantic_count, 2)
        self.assertEqual(result.diagnostics.lexical_count, 2)

    def test_hybrid_retrieve_uses_runtime_candidate_and_bm25_settings(self) -> None:
        calls: dict[str, object] = {}

        def fake_query_index(*args, **_kwargs) -> list[tuple[float, ChunkRecord]]:
            calls["semantic_top_k"] = args[2]
            return [(0.91, self.semantic)]

        def fake_search_lexical_index(*args, **kwargs) -> list[tuple[float, ChunkRecord]]:
            calls["lexical_top_k"] = args[2]
            calls["bm25_k1"] = kwargs["bm25_k1"]
            calls["bm25_b"] = kwargs["bm25_b"]
            return [(12.0, self.lexical)]

        hybrid_retrieval.hybrid_retrieve(
            "support",
            [1.0, 0.0, 0.0],
            "support question",
            top_k=3,
            min_semantic_score=0.2,
            semantic_search=fake_query_index,
            lexical_search=fake_search_lexical_index,
            retrieval_config=RetrievalConfig(
                candidate_multiplier=5,
                candidate_min=10,
                bm25_k1=2.0,
                bm25_b=0.5,
            ),
        )

        self.assertEqual(calls["semantic_top_k"], 15)
        self.assertEqual(calls["lexical_top_k"], 15)
        self.assertEqual(calls["bm25_k1"], 2.0)
        self.assertEqual(calls["bm25_b"], 0.5)

    def test_semantic_mode_skips_lexical_search_and_rrf(self) -> None:
        def fake_query_index(*_args, **_kwargs) -> list[tuple[float, ChunkRecord]]:
            return [(0.91, self.semantic), (0.90, self.both)]

        def unexpected_lexical_search(
            *_args,
            **_kwargs,
        ) -> list[tuple[float, ChunkRecord]]:
            self.fail("lexical search should not run in semantic mode")

        result = hybrid_retrieval.hybrid_retrieve(
            "support",
            [1.0, 0.0, 0.0],
            "support question",
            top_k=1,
            min_semantic_score=0.2,
            semantic_search=fake_query_index,
            lexical_search=unexpected_lexical_search,
            retrieval_config=RetrievalConfig(mode="semantic"),
        )

        self.assertEqual([record.doc_id for _, record in result.hits], ["semantic"])
        self.assertEqual(result.diagnostics.lexical_count, 0)

    def test_hybrid_retrieve_falls_back_to_semantic_when_lexical_errors(self) -> None:
        def fake_query_index(*_args, **_kwargs) -> list[tuple[float, ChunkRecord]]:
            return [(0.91, self.semantic), (0.90, self.both)]

        def broken_search_lexical_index(
            *_args,
            **_kwargs,
        ) -> list[tuple[float, ChunkRecord]]:
            raise RuntimeError("lexical index unavailable")

        with patch.object(hybrid_retrieval.logger, "exception"):
            result = hybrid_retrieval.hybrid_retrieve(
                "support",
                [1.0, 0.0, 0.0],
                "support question",
                top_k=3,
                min_semantic_score=0.2,
                semantic_search=fake_query_index,
                lexical_search=broken_search_lexical_index,
            )

        self.assertEqual(
            [record.doc_id for _, record in result.hits],
            ["semantic", "both"],
        )
        self.assertEqual(result.diagnostics.lexical_count, 0)

    def test_hybrid_retrieve_returns_empty_list_when_index_is_absent(self) -> None:
        def missing_query_index(*_args, **_kwargs) -> list[tuple[float, ChunkRecord]]:
            raise FileNotFoundError(
                "Index not built for rag 'missing' "
                "(expected index=/var/lib/chatfleet/faiss/missing/index.faiss)"
            )

        def unexpected_lexical_search(
            *_args,
            **_kwargs,
        ) -> list[tuple[float, ChunkRecord]]:
            self.fail("lexical search should not run when the semantic index is absent")

        with patch.object(hybrid_retrieval.logger, "warning") as warning:
            result = hybrid_retrieval.hybrid_retrieve(
                "missing",
                [1.0, 0.0, 0.0],
                "support question",
                top_k=3,
                min_semantic_score=0.2,
                semantic_search=missing_query_index,
                lexical_search=unexpected_lexical_search,
            )

        self.assertEqual(result.hits, [])
        self.assertTrue(result.diagnostics.index_missing)
        self.assertIn("/var/lib/chatfleet/faiss/missing", result.diagnostics.index_error or "")
        warning.assert_called_once()


if __name__ == "__main__":
    unittest.main()

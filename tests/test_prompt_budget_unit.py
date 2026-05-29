from app.services.chat import (
    CONTEXT_TOKEN_BUDGET,
    HISTORY_TOKEN_BUDGET,
    MIN_CONTEXT_TOKEN_BUDGET,
    OUTPUT_TOKEN_RESERVE,
    PROMPT_TOKEN_BUDGET,
    TOKEN_CHARS_PER_TOKEN,
    _estimate_tokens,
    _truncate_context,
    _truncate_text_to_tokens,
)
from app.services.vectorstore import ChunkRecord


def test_default_prompt_budgets_target_local_32k_models():
    assert PROMPT_TOKEN_BUDGET == 24000
    assert CONTEXT_TOKEN_BUDGET == 18000
    assert HISTORY_TOKEN_BUDGET == 2500
    assert OUTPUT_TOKEN_RESERVE == 6144
    assert MIN_CONTEXT_TOKEN_BUDGET == 1500
    assert TOKEN_CHARS_PER_TOKEN == 4


def test_truncate_text_to_token_budget():
    text = "x" * 400

    truncated = _truncate_text_to_tokens(text, 25)

    assert _estimate_tokens(truncated) <= 25
    assert len(truncated) < len(text)


def test_truncate_context_respects_token_budget():
    hits = [
        (
            0.9,
            ChunkRecord(
                doc_id=f"doc-{idx}",
                filename="doc.txt",
                chunk_index=idx,
                text="x" * 400,
            ),
        )
        for idx in range(4)
    ]

    kept = list(_truncate_context(hits, budget=150))

    assert len(kept) == 1
    assert sum(_estimate_tokens(record.text) for _, record in kept) <= 150

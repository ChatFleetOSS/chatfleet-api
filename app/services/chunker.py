# generated-by: codex-agent 2025-02-15T00:45:00Z
"""
Document chunking logic used during indexing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Sequence


def _split_paragraphs(text: str) -> List[str]:
    paragraphs = [para.strip() for para in re.split(r"\n\s*\n", text) if para.strip()]
    if paragraphs:
        return paragraphs
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines or [text]


@dataclass(frozen=True)
class PageText:
    """Lightweight page payload produced during PDF extraction."""

    page: int
    text: str


@dataclass(frozen=True)
class ChunkWithPage:
    """Chunk of text with page provenance for downstream citations."""

    text: str
    page_start: int
    page_end: int


def _estimate_tokens(text: str) -> int:
    """Quick-and-dirty token estimator (≈4 characters per token)."""
    text = text.strip()
    if not text:
        return 0
    return max(1, len(text) // 4)


def chunk_pdf(
    pages: Sequence[PageText],
    target_tokens: int = 600,
    overlap_tokens: int = 90,
) -> List[ChunkWithPage]:
    """
    Chunk PDF pages into overlapping windows with page provenance.

    Defaults aim for ~350-400 token chunks with ~15% overlap to balance recall
    and answer quality. Page metadata is preserved to power accurate citations.
    """

    if not pages:
        return []

    target_tokens = max(1, target_tokens)
    overlap_tokens = max(1, overlap_tokens)
    target_chars = target_tokens * 4
    overlap_chars = overlap_tokens * 4
    chunks: List[ChunkWithPage] = []
    buffer = ""
    start_page = pages[0].page
    end_page = start_page
    seen: set[str] = set()

    for page in pages:
        if not page.text.strip():
            continue
        for paragraph in _split_paragraphs(page.text):
            if not paragraph.strip():
                continue
            candidate = f"{buffer}\n\n{paragraph}" if buffer else paragraph
            token_estimate = _estimate_tokens(candidate)

            if (
                token_estimate <= target_tokens * 1.1
                and len(candidate) <= target_chars * 1.1
            ):
                buffer = candidate
                end_page = page.page
                continue

            if buffer:
                normalized = buffer.strip()
                if normalized and normalized not in seen:
                    chunks.append(
                        ChunkWithPage(
                            text=normalized,
                            page_start=start_page,
                            page_end=end_page,
                        )
                    )
                    seen.add(normalized)
                tail = buffer[-overlap_chars:].strip()
                buffer = f"{tail}\n\n{paragraph}".strip()
                start_page = end_page
                end_page = page.page
            else:
                stride = max(target_chars - overlap_chars, 1)
                for idx in range(0, len(paragraph), stride):
                    window = paragraph[idx : idx + target_chars].strip()
                    if not window or window in seen:
                        continue
                    chunks.append(
                        ChunkWithPage(
                            text=window,
                            page_start=page.page,
                            page_end=page.page,
                        )
                    )
                    seen.add(window)
                buffer = ""
                start_page = page.page
                end_page = page.page

    if buffer:
        normalized = buffer.strip()
        if normalized and normalized not in seen:
            chunks.append(
                ChunkWithPage(
                    text=normalized,
                    page_start=start_page,
                    page_end=end_page,
                )
            )

    return chunks


def chunk_text(
    text: str,
    target_chars: int = 1200,
    overlap_chars: int = 200,
) -> List[str]:
    """
    Chunk plain text into overlapping windows (legacy helper).

    Wraps `chunk_pdf` to reuse the balanced window sizing while returning raw
    strings for backwards compatibility with test helpers.
    """

    if not text.strip():
        return []

    approx_tokens = max(1, target_chars // 4)
    approx_overlap = max(1, overlap_chars // 4)
    chunks = chunk_pdf(
        [PageText(page=1, text=text)],
        target_tokens=approx_tokens,
        overlap_tokens=approx_overlap,
    )
    return [chunk.text for chunk in chunks]

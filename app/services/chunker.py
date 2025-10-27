# generated-by: codex-agent 2025-02-15T00:45:00Z
"""
Document chunking logic used during indexing.
"""

from __future__ import annotations

import re
from typing import List


def _split_paragraphs(text: str) -> List[str]:
    paragraphs = [para.strip() for para in re.split(r"\n\s*\n", text) if para.strip()]
    if paragraphs:
        return paragraphs
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines or [text]


def chunk_text(
    text: str,
    target_chars: int = 1200,
    overlap_chars: int = 200,
) -> List[str]:
    """
    Chunk text into overlapping windows of roughly `target_chars`.

    The overlap maintains semantic continuity when chunks are embedded.
    """

    if not text.strip():
        return []

    chunks: List[str] = []
    current = ""

    for paragraph in _split_paragraphs(text):
        candidate = f"{current}\n\n{paragraph}" if current else paragraph
        if len(candidate) <= target_chars:
            current = candidate
            continue

        if current:
            chunks.append(current.strip())
            tail = current[-overlap_chars:].strip()
            current = f"{tail}\n\n{paragraph}".strip()
        else:
            # Paragraph longer than target: hard-split with stride.
            for idx in range(0, len(paragraph), target_chars - overlap_chars):
                window = paragraph[idx : idx + target_chars]
                if window:
                    chunks.append(window.strip())
            current = ""

    if current:
        chunks.append(current.strip())

    # Deduplicate accidental empties
    return [chunk for chunk in chunks if chunk]

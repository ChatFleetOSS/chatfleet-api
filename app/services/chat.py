# generated-by: codex-agent 2025-02-15T00:21:00Z
"""
Chat orchestration with retrieval + LLM-backed answer generation.
"""

from __future__ import annotations

import asyncio
import json
import re
from typing import Any, AsyncGenerator, Dict, List, Sequence, Tuple

from fastapi import status

from app.core.config import settings
from app.core.corr_id import get_corr_id
from app.models.chat import ChatRequest, ChatResponse, Citation, Usage
from app.services.embeddings import embed_text
from app.services.jobs import JobRecord, job_manager
from app.services.llm import generate_chat_completion
from app.services.logging import write_system_log
from app.services.rags import get_rag_by_slug
from app.services.vectorstore import ChunkRecord, query_index
from app.utils.responses import raise_http_error


def _format_hits_for_prompt(hits: Sequence[tuple[float, ChunkRecord]]) -> str:
    if not hits:
        return "No supporting snippets were retrieved from the knowledge base."
    lines: List[str] = []
    for idx, (_, record) in enumerate(hits[:5], start=1):
        snippet = record.text.replace("\n", " ").strip()
        snippet = snippet[:500] + ("…" if len(snippet) > 500 else "")
        lines.append(f"Source {idx} (doc_id={record.doc_id}, file={record.filename}): {snippet}")
    return "\n".join(lines)


def _fallback_answer(question: str, hits: Sequence[tuple[float, ChunkRecord]]) -> str:
    if not hits:
        return (
            "### Résultat indisponible\n\n"
            "Je n'ai trouvé aucun document indexé correspondant à cette question. "
            "Ajoutez ou indexez des documents puis relancez la recherche."
        )
    record = hits[0][1]
    snippet = record.text.replace("\n", " ").strip()
    snippet = snippet[:360] + ("…" if len(snippet) > 360 else "")
    rows: List[str] = ["| Source | Extrait clé |", "| --- | --- |"]
    for _, doc in hits[:3]:
        shortened = doc.text.replace("\n", " ").strip()
        shortened = shortened[:200] + ("…" if len(shortened) > 200 else "")
        safe_text = shortened.replace("|", r"\|")
        rows.append(f"| `{doc.filename}` | {safe_text} |")
    table = "\n".join(rows)
    answer = (
        f"### Synthèse\n\n"
        f"**Question :** {question}\n\n"
        f"{snippet}\n\n"
        "#### Extraits pertinents\n\n"
        f"{table}\n"
    )
    return _normalize_markdown_tables(answer)


def _normalize_markdown_tables(text: str) -> str:
    """Ensure Markdown tables have proper newlines and blank lines."""

    # Insert newlines between adjacent table rows that were glued together.
    fixed = re.sub(r"\|\s*\|\s*(?=\|)", "|\n|", text)

    lines = fixed.splitlines()
    normalized: List[str] = []
    previous_blank = True
    previous_table = False

    for line in lines:
        stripped = line.strip()
        is_table_row = stripped.startswith("|") and stripped.endswith("|")
        if is_table_row:
            if not previous_blank and not previous_table:
                normalized.append("")
            normalized.append(line)
            previous_blank = False
            previous_table = True
            continue

        previous_table = False
        normalized.append(line)
        previous_blank = stripped == ""

    normalized_text = "\n".join(normalized)
    normalized_text = _strip_sources_section(normalized_text)
    normalized_text = _ensure_paragraph_spacing(normalized_text)

    # Preserve the exact number of trailing newlines from the original text.
    trailing_newlines = len(text) - len(text.rstrip("\n"))
    if trailing_newlines:
        normalized_text = normalized_text.rstrip("\n") + ("\n" * trailing_newlines)
    return normalized_text


def _split_markdown_segments(answer: str, max_length: int = 800) -> List[str]:
    """Split markdown into segments respecting newline boundaries."""

    segments: List[str] = []
    buffer = ""

    for line in answer.splitlines(keepends=True):
        if buffer and len(buffer) + len(line) > max_length:
            segments.append(buffer)
            buffer = ""
        buffer += line
    if buffer:
        segments.append(buffer)
    return segments


def _strip_sources_section(text: str) -> str:
    """Remove trailing 'Sources' sections or inline source lines."""

    lines = text.splitlines()
    result: List[str] = []
    skipping = False
    pattern_heading = re.compile(r"^#{1,6}\s*Sources\b", flags=re.IGNORECASE)

    for line in lines:
        stripped = line.strip()
        if not skipping and (
            pattern_heading.match(stripped) or stripped.lower().startswith("sources:")
        ):
            skipping = True
            continue
        if skipping:
            continue
        result.append(line)

    return "\n".join(result)


def _ensure_paragraph_spacing(text: str) -> str:
    """Ensure single blank lines between paragraphs and blocks without disturbing tables."""

    def classify(line: str) -> str:
        stripped = line.lstrip()
        if not stripped:
            return "blank"
        if stripped.startswith("|") and stripped.endswith("|"):
            return "table"
        if bool(re.match(r"\d+\.\s", stripped)):
            return "olist"
        if stripped.startswith(("-", "*", "+")):
            return "ulist"
        if stripped.startswith((">", "`", "~")):
            return "block"
        if stripped.startswith("#"):
            return "heading"
        return "paragraph"

    lines = text.splitlines()
    result: List[str] = []
    prev_type = "blank"

    for line in lines:
        line_type = classify(line)
        if line_type == "blank":
            if result and result[-1] == "":
                continue
            result.append("")
            prev_type = "blank"
            continue

        if result and result[-1] != "":
            if line_type == "table" and prev_type != "table":
                result.append("")
            elif line_type == "paragraph" and prev_type != "paragraph":
                result.append("")
            elif line_type in {"ulist", "olist"} and prev_type not in {line_type, "blank"}:
                result.append("")
            elif line_type in {"heading", "block"} and prev_type not in {line_type, "blank"}:
                result.append("")
        result.append(line)
        prev_type = line_type

    # Collapse multiple trailing blanks
    while result and result[-1] == "":
        result.pop()

    return "\n".join(result)


def _build_prompt_messages(request: ChatRequest, hits: Sequence[tuple[float, ChunkRecord]]) -> List[Dict[str, str]]:
    context = _format_hits_for_prompt(hits)
    system_messages: List[Dict[str, str]] = [
        {
            "role": "system",
            "content": (
                "You are ChatFleet, an assistant that answers with factual precision. "
                "Rely only on the provided knowledge snippets. If the snippets do not "
                "contain the answer, say you are unsure."
            ),
        },
        {
            "role": "system",
            "content": (
                "Knowledge snippets (do not quote outside of these sources):\n"
                f"{context}"
            ),
        },
        {
            "role": "system",
            "content": (
                "Respond in GitHub-flavored Markdown. Use short sections, bullet lists, and tables "
                "to present data clearly. When emitting tables, insert a single blank line before the block, keep rows contiguous "
                "with one newline per row (header, separator, and body rows), and keep cells pipe-separated. "
                "Use a single blank line between paragraphs, and avoid extra blank lines inside tables or lists. "
                "Do not include a dedicated 'Sources' section; supporting material is sent separately."
            ),
        },
    ]

    # Preserve the last turns of the conversation (up to 10 messages).
    history: List[Dict[str, str]] = []
    for message in request.messages[-10:]:
        history.append({"role": message.role, "content": message.content})

    return system_messages + history


def _build_citations_from_hits(hits: Sequence[tuple[float, ChunkRecord]]) -> List[Citation]:
    citations: List[Citation] = []
    seen: set[str] = set()
    for _, record in hits:
        if record.doc_id in seen:
            continue
        seen.add(record.doc_id)
        snippet = record.text.replace("\n", " ")
        snippet = snippet[:280] + ("…" if len(snippet) > 280 else "")
        citations.append(
            Citation(
                doc_id=record.doc_id,
                filename=record.filename,
                pages=[record.chunk_index + 1],
                snippet=snippet,
            )
        )
        if len(citations) >= 3:
            break
    return citations


async def _retrieve_hits(rag_slug: str, question: str, top_k: int) -> List[tuple[float, ChunkRecord]]:
    vector = await embed_text(question)
    loop = asyncio.get_running_loop()

    def _query() -> List[tuple[float, ChunkRecord]]:
        return query_index(rag_slug, vector, top_k)

    try:
        return await loop.run_in_executor(None, _query)
    except FileNotFoundError:
        return []


async def _generate_answer_with_job(
    request: ChatRequest,
    hits: Sequence[tuple[float, ChunkRecord]],
) -> Tuple[str, int]:
    loop = asyncio.get_running_loop()
    future: asyncio.Future[Tuple[str, int]] = loop.create_future()

    async def runner(job: JobRecord) -> None:
        try:
            answer, tokens_out = await _generate_answer(request, hits)
            job.result = {"answer": answer, "tokens_out": tokens_out}
            if not future.done():
                future.set_result((answer, tokens_out))
        except Exception as exc:  # pragma: no cover - defensive
            job.status = "error"
            job.error = str(exc)
            if not future.done():
                future.set_exception(exc)
            raise

    job_manager.schedule("CHAT_COMPLETION", runner)
    return await future


async def _generate_answer(
    request: ChatRequest,
    hits: Sequence[tuple[float, ChunkRecord]],
) -> Tuple[str, int]:
    question = request.messages[-1].content if request.messages else ""
    messages = _build_prompt_messages(request, hits)
    temperature = (
        request.opts.temperature
        if request.opts and request.opts.temperature is not None
        else settings.temperature_default
    )
    max_tokens = (
        request.opts.max_tokens if request.opts and request.opts.max_tokens else 500
    )

    llm_result = await generate_chat_completion(
        messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    if llm_result is not None:
        answer_text, tokens_out = llm_result
        answer_text = _normalize_markdown_tables(answer_text)
        if answer_text:
            return answer_text, tokens_out

    fallback = _fallback_answer(question, hits)
    estimated_tokens = max(1, len(fallback.split()))
    return fallback, estimated_tokens


async def handle_chat(request: ChatRequest, user_id: str) -> ChatResponse:
    rag = await get_rag_by_slug(request.rag_slug)
    if not rag:
        raise_http_error("RAG_NOT_FOUND", f"RAG '{request.rag_slug}' not found", status_code=404)

    top_k = request.opts.top_k if request.opts and request.opts.top_k else settings.top_k_default
    last_message = request.messages[-1]
    hits = await _retrieve_hits(request.rag_slug, last_message.content, top_k)
    try:
        answer, tokens_out = await _generate_answer_with_job(request, hits)
    except Exception:
        raise_http_error("CHAT_FAILED", "Unable to generate an answer at this time", status.HTTP_503_SERVICE_UNAVAILABLE)

    citations = _build_citations_from_hits(hits)
    usage = Usage(
        tokens_in=sum(len(msg.content) for msg in request.messages),
        tokens_out=tokens_out,
    )
    corr_id = get_corr_id()
    response = ChatResponse(answer=answer, citations=citations, usage=usage, corr_id=corr_id)

    await write_system_log(
        event="chat.completion",
        rag_slug=request.rag_slug,
        user_id=user_id,
        details={
            "message_count": len(request.messages),
            "tokens_out": usage.tokens_out,
        },
        level="info",
    )
    return response


async def stream_chat(request: ChatRequest, user_id: str) -> AsyncGenerator[str, None]:
    rag = await get_rag_by_slug(request.rag_slug)
    if not rag:
        raise_http_error("RAG_NOT_FOUND", f"RAG '{request.rag_slug}' not found", status_code=404)

    corr_id = get_corr_id()
    question = request.messages[-1].content
    top_k = request.opts.top_k if request.opts and request.opts.top_k else settings.top_k_default
    hits = await _retrieve_hits(request.rag_slug, question, top_k)
    try:
        answer, tokens_out = await _generate_answer_with_job(request, hits)
    except Exception:
        raise_http_error("CHAT_FAILED", "Unable to generate an answer at this time", status.HTTP_503_SERVICE_UNAVAILABLE)

    segments = _split_markdown_segments(answer)
    if not segments:
        segments = [answer]

    async def send(event: str, payload: Any) -> str:
        if payload is None:
            data: Any = {}
        elif isinstance(payload, dict):
            data = dict(payload)
        else:
            data = payload

        if isinstance(data, dict) and event != "ping":
            data.setdefault("corr_id", corr_id)
        return f"event: {event}\ndata: {json.dumps(data)}\n\n"

    yield await send("ready", {"corr_id": corr_id})
    for chunk in segments:
        await asyncio.sleep(0.05)
        yield await send("chunk", {"delta": chunk})
    await asyncio.sleep(0.05)
    citations = [citation.model_dump() for citation in _build_citations_from_hits(hits)]
    yield await send("citations", {"citations": citations})
    usage = {
        "tokens_in": sum(len(msg.content) for msg in request.messages),
        "tokens_out": tokens_out,
    }
    yield await send("done", {"usage": usage})
    yield await send("ping", {})

    await write_system_log(
        event="chat.stream",
        rag_slug=request.rag_slug,
        user_id=user_id,
        details={"corr_id": corr_id, "chunks": len(segments)},
    )

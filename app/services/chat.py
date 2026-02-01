# generated-by: codex-agent 2025-02-15T00:21:00Z
"""
Chat orchestration with retrieval + LLM-backed answer generation.
"""

from __future__ import annotations

import asyncio
import json
import re
import logging
from typing import Any, AsyncGenerator, Dict, List, Sequence, Tuple

from fastapi import status

from app.core.config import settings
from app.core.corr_id import get_corr_id
from app.models.chat import ChatRequest, ChatResponse, Citation, Usage
from app.services.embeddings import embed_text
from app.services.jobs import JobRecord, job_manager
from app.services.llm import generate_chat_completion
from app.services.runtime_config import get_llm_config, get_api_key
from app.services.logging import write_system_log
from app.services.rags import get_rag_by_slug
from app.services.vectorstore import ChunkRecord, query_index
from app.utils.responses import raise_http_error

logger = logging.getLogger("chatfleet.chat")
logger.setLevel(logging.INFO)
retrieval_logger = logging.getLogger("chatfleet.retrieval")
retrieval_logger.setLevel(logging.INFO)
prompt_logger = logging.getLogger("chatfleet.prompt")
prompt_logger.setLevel(logging.INFO)
RETRIEVAL_MIN_SCORE = 0.2
CONTEXT_CHAR_BUDGET = 6000
FALLBACK_ANSWER = "Je n'ai pas cette information dans les extraits fournis."


def _format_hits_for_prompt(hits: Sequence[tuple[float, ChunkRecord]]) -> str:
    if not hits:
        return "No supporting snippets were retrieved from the knowledge base."
    lines: List[str] = []
    for idx, (score, record) in enumerate(hits[:6], start=1):
        page_hint = ""
        if record.page_start:
            if record.page_end and record.page_end != record.page_start:
                page_hint = f" pages {record.page_start}-{record.page_end}"
            else:
                page_hint = f" page {record.page_start}"
        source = (
            f"[{idx}] score={score:.3f} doc_id={record.doc_id} file={record.filename}{page_hint}\n"
            f"{record.text.strip()}"
        )
        lines.append(source)
    return "\n\n".join(lines)


def _preview_hits(hits: Sequence[tuple[float, ChunkRecord]], limit: int = 3) -> List[Dict[str, Any]]:
    previews: List[Dict[str, Any]] = []
    for score, record in hits[:limit]:
        snippet = record.text.replace("\n", " ").strip()
        snippet = snippet[:180] + ("…" if len(snippet) > 180 else "")
        previews.append(
            {
                "score": float(score),
                "doc_id": record.doc_id,
                "file": record.filename,
                "page_start": record.page_start,
                "page_end": record.page_end,
                "chunk_index": record.chunk_index,
                "snippet": snippet,
            }
        )
    return previews


def _truncate_context(hits: Sequence[tuple[float, ChunkRecord]], budget: int = CONTEXT_CHAR_BUDGET) -> Sequence[tuple[float, ChunkRecord]]:
    """Limit total context characters to avoid oversized prompts."""
    total = 0
    kept: List[tuple[float, ChunkRecord]] = []
    for pair in hits:
        text_len = len(pair[1].text)
        if kept and total + text_len > budget:
            break
        kept.append(pair)
        total += text_len
    return kept


def _topk_log_text(hits: Sequence[tuple[float, ChunkRecord]], limit: int = 10) -> str:
    parts: List[str] = []
    for idx, (score, record) in enumerate(hits[:limit], start=1):
        page_hint = ""
        if record.page_start:
            if record.page_end and record.page_end != record.page_start:
                page_hint = f" pages {record.page_start}-{record.page_end}"
            else:
                page_hint = f" page {record.page_start}"
        parts.append(
            f"{idx}) score={score:.3f} doc={record.doc_id} file={record.filename}{page_hint} idx={record.chunk_index} len={len(record.text)}\n{record.text.strip()}"
        )
    return "\n".join(parts)


def _has_query_overlap(question: str, hits: Sequence[tuple[float, ChunkRecord]]) -> bool:
    """Check if at least one query token appears in the retrieved snippets."""
    tokens = set(re.findall(r"\b\w{4,}\b", question.lower()))
    if not tokens:
        return False
    for _, record in hits:
        text = record.text.lower()
        if any(tok in text for tok in tokens):
            return True
    return False


def _log_prompt_messages(messages: List[Dict[str, str]], rag_slug: str) -> None:
    corr_id = get_corr_id()
    try:
        preview: List[str] = []
        for idx, msg in enumerate(messages):
            content = msg.get("content", "")
            preview.append(f"[{idx}:{msg.get('role','?')}] {content}")
        prompt_logger.info(
            "chat.prompt.final corr_id=%s rag=%s count=%s\n%s",
            corr_id,
            rag_slug,
            len(messages),
            "\n".join(preview),
            extra={
                "corr_id": corr_id,
                "rag_slug": rag_slug,
                "message_count": len(messages),
                "prompt_messages": messages,
            },
        )
    except Exception:
        pass


class LLMUnavailableError(Exception):
    pass


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
    pattern_unavailable = re.compile(r"^\*?\s*Sources\s+indisponibles\.?\s*\*?$", flags=re.IGNORECASE)

    for line in lines:
        stripped = line.strip()
        if pattern_unavailable.match(stripped):
            continue
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
                "You are a helpful and warm documentation assistant.\n\n"
                "GROUNDING (ABSOLUTE RULE)\n"
                "- You must base your answers exclusively on the provided context excerpts.\n"
                "- Do NOT add prior knowledge, assumptions, or general explanations.\n"
                "- Do NOT invent definitions or mechanisms not supported by the excerpts.\n\n"
                "USING THE CONTEXT\n"
                "- You are allowed and encouraged to combine information from multiple excerpts.\n"
                "- You may synthesize, reorganize, and rephrase the excerpts to give a clear explanation.\n"
                "- When a concept is spread across several excerpts, explain it globally.\n"
                "- If the question is broad or if the user simply provides a topic or title, interpret it as a request for all relevant information and provide an organized overview based on the available excerpts.\n\n"
                "ANSWER STYLE\n"
                "- Be clear, calm, and pedagogical.\n"
                "- Do NOT be overly brief.\n"
                "- Prefer a structured explanation over a minimal answer.\n"
                "- When relevant and supported by the excerpts, explain: what the concept is, how it works, what it covers or applies to.\n\n"
                "STRUCTURE (when possible)\n"
                "1. A short summary sentence\n"
                "2. A structured explanation (paragraphs or bullet points)\n\n"
                "FORMATTING & TONE\n"
                "- Do not mention sources, chunk IDs, or retrieval details.\n"
                "- Respond in the user’s language.\n"
                "- Use GitHub-flavored Markdown.\n"
                "- Friendly, neutral, professional tone.\n"
                "- If the excerpts do not contain anything relevant, say so plainly."
            ),
        },
        {
            "role": "system",
            "content": (
                "Context:\n"
                f"{context}"
            ),
        },
    ]

    # Preserve the last turns of the conversation (up to 10 messages).
    history: List[Dict[str, str]] = []
    for message in request.messages[-10:]:
        history.append({"role": message.role, "content": message.content})

    try:
        logger.info(
            "chat.prompt",
            extra={
                "corr_id": get_corr_id(),
                "system_count": len(system_messages),
                "history_count": len(history),
                "context_preview": context[:500],
            },
        )
    except Exception:
        pass

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
        if record.page_start:
            end = record.page_end or record.page_start
            pages = list(range(record.page_start, end + 1))
        else:
            pages = [record.chunk_index + 1]
        citations.append(
            Citation(
                doc_id=record.doc_id,
                filename=record.filename,
                pages=pages,
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
        return query_index(rag_slug, vector, top_k, min_score=RETRIEVAL_MIN_SCORE)

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
    _log_prompt_messages(messages, request.rag_slug)
    # prefer runtime default
    cfg = await get_llm_config()
    temperature = (
        request.opts.temperature
        if request.opts and request.opts.temperature is not None
        else cfg.temperature_default
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
    # No LLM available or failed
    raise LLMUnavailableError("LLM_UNAVAILABLE")


async def _ensure_llm_configured() -> bool:
    cfg = await get_llm_config()
    if cfg.provider == "openai":
        key = await get_api_key()
        import os
        return bool(key or os.getenv("OPENAI_API_KEY"))
    if cfg.provider == "vllm":
        return bool(cfg.base_url)
    return False


async def handle_chat(request: ChatRequest, user_id: str) -> ChatResponse:
    if not await _ensure_llm_configured():
        raise_http_error(
            "LLM_NOT_CONFIGURED",
            "LLM provider is not configured. In Admin → Settings, choose OpenAI or vLLM, provide the API key or base URL, then Save. After changing the embedding model, rebuild indexes for best results.",
            status.HTTP_503_SERVICE_UNAVAILABLE,
        )
    rag = await get_rag_by_slug(request.rag_slug)
    if not rag:
        raise_http_error("RAG_NOT_FOUND", f"RAG '{request.rag_slug}' not found", status_code=404)

    cfg = await get_llm_config()
    top_k = request.opts.top_k if request.opts and request.opts.top_k else cfg.top_k_default
    last_message = request.messages[-1]
    question = last_message.content
    hits = await _retrieve_hits(request.rag_slug, question, top_k)
    if not hits:
        raise_http_error(
            "NO_CONTEXT",
            "No supporting snippets were retrieved for this query; cannot answer without context.",
            status.HTTP_503_SERVICE_UNAVAILABLE,
        )
    if not _has_query_overlap(question, hits):
        usage = Usage(
            tokens_in=sum(len(msg.content) for msg in request.messages),
            tokens_out=0,
        )
        corr_id = get_corr_id()
        return ChatResponse(answer=FALLBACK_ANSWER, citations=[], usage=usage, corr_id=corr_id)
    context_hits = list(_truncate_context(hits))
    corr_id = get_corr_id()
    retrieval_logger.info(
        "chat.retrieval",
        extra={
            "corr_id": corr_id,
            "rag_slug": request.rag_slug,
            "top_k": top_k,
            "min_score": RETRIEVAL_MIN_SCORE,
            "hit_count": len(hits),
            "hits": [
                {
                    "score": float(score),
                    "doc_id": record.doc_id,
                    "chunk_index": record.chunk_index,
                    "filename": record.filename,
                    "page_start": record.page_start,
                    "page_end": record.page_end,
                }
                for score, record in hits[:10]
            ],
            "hit_previews": _preview_hits(hits),
            "question": last_message.content,
            "context_char_budget": CONTEXT_CHAR_BUDGET,
        },
    )
    try:
        retrieval_logger.info(
            "chat.retrieval.topk corr_id=%s rag=%s %s",
            corr_id,
            request.rag_slug,
            _topk_log_text(hits, limit=top_k),
        )
    except Exception:
        pass
    try:
        prompt_logger.info(
            "chat.prompt.built",
            extra={
                "corr_id": corr_id,
                "rag_slug": request.rag_slug,
                "question": last_message.content,
                "top_k": top_k,
                "context_preview": _format_hits_for_prompt(context_hits)[:1000],
            },
        )
    except Exception:
        pass
    try:
        answer, tokens_out = await _generate_answer_with_job(request, context_hits)
    except LLMUnavailableError:
        raise_http_error(
            "LLM_UNAVAILABLE",
            "LLM provider is unreachable or returned no completion. Verify the provider URL/model and try again.",
            status.HTTP_503_SERVICE_UNAVAILABLE,
        )
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
    if not await _ensure_llm_configured():
        raise_http_error(
            "LLM_NOT_CONFIGURED",
            "LLM provider is not configured. In Admin → Settings, choose OpenAI or vLLM, provide the API key or base URL, then Save. After changing the embedding model, rebuild indexes for best results.",
            status.HTTP_503_SERVICE_UNAVAILABLE,
        )
    rag = await get_rag_by_slug(request.rag_slug)
    if not rag:
        raise_http_error("RAG_NOT_FOUND", f"RAG '{request.rag_slug}' not found", status_code=404)

    corr_id = get_corr_id()
    question = request.messages[-1].content
    cfg = await get_llm_config()
    top_k = request.opts.top_k if request.opts and request.opts.top_k else cfg.top_k_default
    hits = await _retrieve_hits(request.rag_slug, question, top_k)
    if not hits:
        async def _send_error() -> AsyncGenerator[str, None]:
            payload = {
                "error": {
                    "code": "NO_CONTEXT",
                    "message": "No supporting snippets were retrieved for this query; cannot answer without context.",
                },
                "corr_id": corr_id,
            }
            yield f"event: error\ndata: {json.dumps(payload)}\n\n"
            yield f"event: done\ndata: {json.dumps({'usage': {'tokens_in': 0, 'tokens_out': 0}, 'corr_id': corr_id})}\n\n"
        async for chunk in _send_error():
            yield chunk
        return
    if not _has_query_overlap(question, hits):
        async def _send_fallback() -> AsyncGenerator[str, None]:
            usage = {"tokens_in": sum(len(msg.content) for msg in request.messages), "tokens_out": 0}
            payload_ready = {"corr_id": corr_id}
            yield f"event: ready\ndata: {json.dumps(payload_ready)}\n\n"
            yield f"event: chunk\ndata: {json.dumps({'delta': FALLBACK_ANSWER, 'corr_id': corr_id})}\n\n"
            yield f"event: citations\ndata: {json.dumps({'citations': [], 'corr_id': corr_id})}\n\n"
            yield f"event: done\ndata: {json.dumps({'usage': usage, 'corr_id': corr_id})}\n\n"
            yield f"event: ping\ndata: {json.dumps({'corr_id': corr_id})}\n\n"
        async for chunk in _send_fallback():
            yield chunk
        await write_system_log(
            event="chat.stream",
            rag_slug=request.rag_slug,
            user_id=user_id,
            details={"corr_id": corr_id, "chunks": 1, "fallback": True},
        )
        return
    context_hits = list(_truncate_context(hits))
    logger.info(
        "chat.retrieval.stream",
        extra={
            "corr_id": corr_id,
            "rag_slug": request.rag_slug,
            "top_k": top_k,
            "min_score": RETRIEVAL_MIN_SCORE,
            "hit_count": len(hits),
            "hits": [
                {
                    "score": float(score),
                    "doc_id": record.doc_id,
                    "chunk_index": record.chunk_index,
                    "filename": record.filename,
                    "page_start": record.page_start,
                    "page_end": record.page_end,
                }
                for score, record in hits[:10]
            ],
            "hit_previews": _preview_hits(hits),
            "question": question,
            "context_char_budget": CONTEXT_CHAR_BUDGET,
        },
    )
    try:
        logger.info(
            "chat.retrieval.topk corr_id=%s rag=%s %s",
            corr_id,
            request.rag_slug,
            _topk_log_text(hits, limit=top_k),
        )
    except Exception:
        pass
    try:
        prompt_logger.info(
            "chat.prompt.built",
            extra={
                "corr_id": corr_id,
                "rag_slug": request.rag_slug,
                "question": question,
                "top_k": top_k,
                "context_preview": _format_hits_for_prompt(context_hits)[:1000],
            },
        )
    except Exception:
        pass
    try:
        answer, tokens_out = await _generate_answer_with_job(request, context_hits)
    except LLMUnavailableError:
        async def _send_error() -> AsyncGenerator[str, None]:
            payload = {
                "error": {
                    "code": "LLM_UNAVAILABLE",
                    "message": "LLM provider is unreachable or returned no completion. Verify the provider URL/model and try again.",
                },
                "corr_id": corr_id,
            }
            yield f"event: error\ndata: {json.dumps(payload)}\n\n"
            yield f"event: done\ndata: {json.dumps({'usage': {'tokens_in': 0, 'tokens_out': 0}, 'corr_id': corr_id})}\n\n"
        async for chunk in _send_error():
            yield chunk
        return
    except Exception:
        async def _send_error() -> AsyncGenerator[str, None]:
            payload = {
                "error": {
                    "code": "CHAT_FAILED",
                    "message": "Unable to generate an answer at this time.",
                },
                "corr_id": corr_id,
            }
            yield f"event: error\ndata: {json.dumps(payload)}\n\n"
            yield f"event: done\ndata: {json.dumps({'usage': {'tokens_in': 0, 'tokens_out': 0}, 'corr_id': corr_id})}\n\n"
        async for chunk in _send_error():
            yield chunk
        return

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

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from app.dependencies.auth import require_admin
from app.models.admin import (
    LLMConfigResponse,
    LLMConfigUpdateRequest,
    LLMConfigTestRequest,
    LLMConfigTestResult,
    LLMModelsRequest,
    LLMModelsResponse,
    LLMEmbedTestResult,
)
from app.services.runtime_config import get_llm_config, set_llm_config, set_verified
from app.services.llm import test_chat_completion_provider, discover_models
from app.services.embeddings import test_embedding_provider, prewarm_embeddings
from app.utils.responses import with_corr_id

router = APIRouter(prefix="/admin/llm", tags=["Admin LLM Config"])


@router.get("/config", response_model=LLMConfigResponse)
async def get_config(current_user = Depends(require_admin)) -> LLMConfigResponse:
    cfg = await get_llm_config()
    return LLMConfigResponse(**with_corr_id({"config": cfg}))


@router.post("/config/test", response_model=LLMConfigTestResult)
async def test_config(payload: LLMConfigTestRequest, current_user = Depends(require_admin)) -> LLMConfigTestResult:
    ok, message = await test_chat_completion_provider(payload)
    return LLMConfigTestResult(**with_corr_id({"ok": ok, "message": message}))


@router.put("/config", response_model=LLMConfigResponse)
async def put_config(payload: LLMConfigUpdateRequest, current_user = Depends(require_admin)) -> LLMConfigResponse:
    cfg = await set_llm_config(payload, str(current_user["_id"]))
    ok, _ = await test_chat_completion_provider(LLMConfigTestRequest(**payload.model_dump()))
    await set_verified(ok)
    # Best-effort prewarm in background (chat + embeddings)
    import asyncio
    async def _prewarm() -> None:
        try:
            # tiny chat completion
            from app.services.llm import generate_chat_completion
            await generate_chat_completion([
                {"role": "user", "content": "ping"}
            ], temperature=0, max_tokens=1)
        except Exception:
            pass
        try:
            await prewarm_embeddings()
        except Exception:
            pass
    try:
        asyncio.create_task(_prewarm())
    except Exception:
        pass
    return LLMConfigResponse(**with_corr_id({"config": cfg}))


@router.post("/config/models", response_model=LLMModelsResponse)
async def post_models(payload: LLMModelsRequest, current_user = Depends(require_admin)) -> LLMModelsResponse:
    try:
        chat, emb, raw = await discover_models(payload.provider, payload.base_url, payload.api_key)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except TimeoutError as exc:
        raise HTTPException(status_code=status.HTTP_504_GATEWAY_TIMEOUT, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc
    return LLMModelsResponse(**with_corr_id({
        "chat_models": chat,
        "embed_models": emb,
        "raw_models": raw,
    }))


@router.post("/config/test-embed", response_model=LLMEmbedTestResult)
async def test_embed(payload: LLMConfigTestRequest, current_user = Depends(require_admin)) -> LLMEmbedTestResult:
    ok, message, dim = await test_embedding_provider(payload)
    return LLMEmbedTestResult(**with_corr_id({"ok": ok, "message": message, "dim": dim}))

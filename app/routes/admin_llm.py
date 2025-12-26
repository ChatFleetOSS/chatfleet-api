from __future__ import annotations

from fastapi import APIRouter, Depends

from app.dependencies.auth import require_admin
from app.models.admin import (
    LLMConfigResponse,
    LLMConfigUpdateRequest,
    LLMConfigTestRequest,
    LLMConfigTestResult,
)
from app.services.runtime_config import get_llm_config, set_llm_config, set_verified
from app.services.llm import test_chat_completion_provider
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
    return LLMConfigResponse(**with_corr_id({"config": cfg}))


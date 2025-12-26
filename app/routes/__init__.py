# generated-by: codex-agent 2025-02-15T00:30:00Z
"""API router registration."""

from fastapi import FastAPI

from .access import router as access_router
from .admin_config import router as admin_config_router
from .admin_users import router as admin_users_router
from .auth import router as auth_router
from .chat import router as chat_router
from .jobs import router as jobs_router
from .rag import router as rag_router
from .rag_admin import router as rag_admin_router
from .health import router as health_router
from .admin_llm import router as admin_llm_router


def register_routes(app: FastAPI) -> None:
    app.include_router(auth_router, prefix="/api")
    app.include_router(rag_router, prefix="/api")
    app.include_router(rag_admin_router, prefix="/api")
    app.include_router(access_router, prefix="/api")
    app.include_router(chat_router, prefix="/api")
    app.include_router(jobs_router, prefix="/api")
    app.include_router(admin_users_router, prefix="/api")
    app.include_router(admin_config_router, prefix="/api")
    app.include_router(admin_llm_router, prefix="/api")
    app.include_router(health_router, prefix="/api")

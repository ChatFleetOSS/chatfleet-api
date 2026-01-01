# generated-by: codex-agent 2025-02-15T00:30:00Z
"""
FastAPI application entrypoint for the ChatFleet backend (MVP v0.1.1).
"""

from __future__ import annotations

import logging
import sys
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.core.corr_id import CorrIdMiddleware
from app.core.database import get_client
from app.core.mongo_launcher import ensure_local_mongo, stop_local_mongo
from app.routes import register_routes
from app.services.bootstrap import run_startup
from app.utils.error_handlers import install_error_handlers
from app.utils.responses import UTF8JSONResponse

def _configure_logging() -> None:
    """Ensure an INFO-level console handler exists (uvicorn may preconfigure logging)."""
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s - %(message)s")
    root = logging.getLogger()
    if not root.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(formatter)
        root.addHandler(handler)
    root.setLevel(logging.INFO)
    for name in [
        "chatfleet",
        "chatfleet.backend",
        "chatfleet.rag.ingest",
        "chatfleet.vectorstore",
        "chatfleet.embeddings",
        "chatfleet.retrieval",
        "chatfleet.chat",
    ]:
        logging.getLogger(name).setLevel(logging.INFO)


_configure_logging()
logger = logging.getLogger("chatfleet.backend")


def create_app() -> FastAPI:
    app = FastAPI(
        title="ChatFleet API",
        version="0.1.1",
        description="ChatFleet — multi-RAG chatbot platform (MVP v0.1.1)",
        openapi_url="/api/openapi.json",
        default_response_class=UTF8JSONResponse,
    )

    app.add_middleware(CorrIdMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    install_error_handlers(app)
    register_routes(app)

    @app.on_event("startup")
    async def _startup() -> None:
        logger.info("ChatFleet backend starting")
        await ensure_local_mongo()
        await run_startup()

    @app.on_event("shutdown")
    async def _shutdown() -> None:
        logger.info("ChatFleet backend shutting down")
        client = get_client()
        client.close()
        stop_local_mongo()

    return app


app = create_app()

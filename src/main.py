"""
main.py — FastAPI application entrypoint for the Jarvis Evaluation Service.
"""

from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.routes.text_to_sql_evaluation_router import router as eval_router
from src.config.settings import settings

# ── Logging configuration ──────────────────────────────────────────────────────

def _configure_logging() -> None:
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, settings.LOG_LEVEL, logging.INFO),
    )
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.format_exc_info,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.JSONRenderer(),
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )


_configure_logging()
logger = structlog.get_logger("jarvis_eval")


# ── Lifespan ───────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(
        "jarvis_evaluation_service.startup",
        port=settings.PORT,
        env=settings.APP_ENV,
        langfuse_configured=settings.is_langfuse_configured,
        trino_enabled=settings.TRINO_ENABLED,
    )
    yield
    logger.info("jarvis_evaluation_service.shutdown")


# ── FastAPI app ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Jarvis Evaluation Service",
    description=(
        "Generic AI Evaluation Platform.\n\n"
        "Current use case: Text-to-SQL agent evaluation against Langfuse datasets.\n\n"
        "Architecture: Plugin-based evaluator system with clean separation of "
        "domain, application, infrastructure, and API layers."
    ),
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# ── Middleware ─────────────────────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_context_middleware(request: Request, call_next):
    """Bind request-level context to structlog for every request."""
    import uuid

    from structlog.contextvars import bind_contextvars, clear_contextvars

    request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
    bind_contextvars(request_id=request_id, path=request.url.path)
    try:
        response = await call_next(request)
        response.headers["x-request-id"] = request_id
        return response
    finally:
        clear_contextvars()


# ── Global error handler ───────────────────────────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(
        "unhandled_exception",
        path=request.url.path,
        error=str(exc),
        exc_info=True,
    )
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )


# ── Routes ─────────────────────────────────────────────────────────────────────

app.include_router(eval_router)


@app.get("/health", tags=["Health"])
async def health_check() -> dict:
    """Service health check."""
    return {
        "status": "ok",
        "service": "jarvis-evaluation-service",
        "version": "0.1.0",
        "langfuse_configured": settings.is_langfuse_configured,
        "trino_enabled": settings.TRINO_ENABLED,
    }

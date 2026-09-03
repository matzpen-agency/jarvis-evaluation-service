"""
langfuse_client_factory.py — Creates and configures the Langfuse SDK client.
"""

from __future__ import annotations

import os

import langfuse as lf_sdk
import structlog

from src.config.settings import Settings

logger = structlog.get_logger(__name__)


def create_langfuse_client(settings: Settings) -> lf_sdk.Langfuse | None:
    """
    Create and return a Langfuse SDK client.

    Sets the required environment variables so that decorator-based tracing
    (langfuse.decorators.observe) also picks them up automatically.
    Env vars are only set after the client is successfully authenticated
    so the exception path leaves the environment unchanged.

    Returns None if credentials are not configured (graceful degradation).
    """
    if not settings.is_langfuse_configured:
        logger.warning("langfuse_client_factory.no_credentials")
        return None

    public_key = settings.LANGFUSE_PUBLIC_KEY.get_secret_value()
    secret_key = settings.LANGFUSE_SECRET_KEY.get_secret_value()

    try:
        client = lf_sdk.Langfuse(
            public_key=public_key,
            secret_key=secret_key,
            host=settings.LANGFUSE_HOST,
        )
        client.auth_check()
        # Only set env vars after auth succeeds so decorator-based tracing works
        os.environ["LANGFUSE_PUBLIC_KEY"] = public_key
        os.environ["LANGFUSE_SECRET_KEY"] = secret_key
        os.environ["LANGFUSE_HOST"] = settings.LANGFUSE_HOST
        logger.info("langfuse_client_factory.connected", host=settings.LANGFUSE_HOST)
        return client
    except Exception as exc:
        logger.error("langfuse_client_factory.failed", error=str(exc))
        return None

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

    Returns None if credentials are not configured (graceful degradation).
    """
    if not settings.is_langfuse_configured:
        logger.warning("langfuse_client_factory.no_credentials")
        return None

    os.environ["LANGFUSE_PUBLIC_KEY"] = settings.LANGFUSE_PUBLIC_KEY
    os.environ["LANGFUSE_SECRET_KEY"] = settings.LANGFUSE_SECRET_KEY
    os.environ["LANGFUSE_HOST"] = settings.LANGFUSE_HOST

    try:
        client = lf_sdk.Langfuse(
            public_key=settings.LANGFUSE_PUBLIC_KEY,
            secret_key=settings.LANGFUSE_SECRET_KEY,
            host=settings.LANGFUSE_HOST,
        )
        client.auth_check()
        logger.info("langfuse_client_factory.connected", host=settings.LANGFUSE_HOST)
        return client
    except Exception as exc:
        logger.error("langfuse_client_factory.failed", error=str(exc))
        return None

"""
trino_client_factory.py — Creates Trino DBAPI connections.

Supports: no-auth, BasicAuthentication, CertificateAuthentication (mTLS).
Mirrors the pattern from core/src/core/trino.py in text2sql-onboarding.
"""

from __future__ import annotations

import structlog
import trino
import trino.auth

from src.config.settings import Settings

logger = structlog.get_logger(__name__)


def create_trino_connection(settings: Settings) -> trino.dbapi.Connection:
    """
    Create a Trino DBAPI connection from Settings.

    Auth priority:
      1. mTLS (cert_path + key_path)
      2. Basic (password)
      3. No auth
    """
    auth = None

    if settings.TRINO_CERT_PATH and settings.TRINO_KEY_PATH:
        auth = trino.auth.CertificateAuthentication(
            settings.TRINO_CERT_PATH,
            settings.TRINO_KEY_PATH,
        )
        logger.debug("trino_client_factory.using_mtls")
    elif settings.TRINO_PASSWORD:
        auth = trino.auth.BasicAuthentication(
            settings.TRINO_USER,
            settings.TRINO_PASSWORD,
        )
        logger.debug("trino_client_factory.using_basic_auth")

    return trino.dbapi.connect(
        host=settings.TRINO_HOST,
        port=settings.TRINO_PORT,
        user=settings.TRINO_USER,
        catalog=settings.TRINO_CATALOG,
        schema=settings.TRINO_SCHEMA,
        http_scheme=settings.TRINO_HTTP_SCHEME,
        auth=auth,
        request_timeout=settings.TRINO_REQUEST_TIMEOUT,
        verify=settings.TRINO_VERIFY,
    )

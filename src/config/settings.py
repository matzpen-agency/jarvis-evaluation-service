"""
settings.py — Application configuration via pydantic-settings.

All values are read from environment variables or a .env file.
Supports Trino mTLS, basic auth, and no-auth modes.
"""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central settings object — injected throughout the application."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Langfuse ──────────────────────────────────────────────────────────────
    LANGFUSE_PUBLIC_KEY: str = Field(default="", description="Langfuse public API key")
    LANGFUSE_SECRET_KEY: str = Field(default="", description="Langfuse secret API key")
    LANGFUSE_HOST: str = Field(
        default="https://cloud.langfuse.com",
        description="Langfuse server URL",
    )

    # ── Trino ─────────────────────────────────────────────────────────────────
    TRINO_HOST: str = "localhost"
    TRINO_PORT: int = 8080
    TRINO_USER: str = "trino"
    TRINO_PASSWORD: str = ""
    TRINO_CATALOG: str = "minio"
    TRINO_SCHEMA: str = "simple_retail"
    TRINO_HTTP_SCHEME: str = "http"
    TRINO_REQUEST_TIMEOUT: float = 30.0
    TRINO_ENABLED: bool = True
    TRINO_VERIFY: bool | str = False
    TRINO_CERT_PATH: str | None = None
    TRINO_KEY_PATH: str | None = None

    # ── Agent ─────────────────────────────────────────────────────────────────
    AGENT_URL: str = "http://localhost:8000"
    AGENT_ENDPOINT: str = "/api/agent/chat"
    AGENT_TIMEOUT: float = 120.0

    # ── Backend (for resolving production tables and query execution) ─────────
    BACKEND_URL: str = "http://localhost:8000"
    BACKEND_QUERY_ENDPOINT: str = "/api/query/execute"
    BACKEND_TIMEOUT: float = 30.0

    # ── Evaluation ────────────────────────────────────────────────────────────
    MAX_CONCURRENT_EVALUATIONS: int = 2
    EVALUATION_HITL_ENABLED: bool = False

    # Time shift offsets in days (negative = past)
    TIME_SHIFT_OFFSETS_DAYS: list[int] = [-1, -7, -14, -30, -60]

    # Composite score weights — must sum to 1.0
    COMPOSITE_WEIGHT_EXECUTION_ACCURACY: float = 0.60
    COMPOSITE_WEIGHT_CONTAINS_ACCURACY: float = 0.15
    COMPOSITE_WEIGHT_SQL_EXACT_MATCH: float = 0.15
    COMPOSITE_WEIGHT_TIME_SHIFT: float = 0.10
    COMPOSITE_WEIGHT_COMPONENT_MATCH: float = 0.0
    COMPOSITE_WEIGHT_SCHEMA_HALLUCINATION: float = 0.0
    COMPOSITE_WEIGHT_DIALECT_ERROR: float = 0.0

    SPIDER2_QUESTIONS_PATH: str = "src/config/spider2_questions.json"

    # Numeric tolerance for result comparison (decimal places)
    NUMERIC_COMPARISON_TOLERANCE: int = 6

    # ── Service ───────────────────────────────────────────────────────────────
    PORT: int = 5002
    APP_ENV: str = "development"
    LOG_LEVEL: str = "INFO"

    @property
    def agent_chat_url(self) -> str:
        """Full agent chat endpoint URL."""
        return f"{self.AGENT_URL}{self.AGENT_ENDPOINT}"

    @property
    def is_langfuse_configured(self) -> bool:
        """True when Langfuse credentials are present."""
        return bool(self.LANGFUSE_PUBLIC_KEY and self.LANGFUSE_SECRET_KEY)


settings = Settings()

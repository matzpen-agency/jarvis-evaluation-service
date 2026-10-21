"""
settings.py — Application configuration via pydantic-settings.

All values are read from environment variables or a .env file.
Supports Trino mTLS, basic auth, and no-auth modes.
"""

from __future__ import annotations

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central settings object — injected throughout the application."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Langfuse ──────────────────────────────────────────────────────────────
    LANGFUSE_PUBLIC_KEY: SecretStr = Field(default=SecretStr(""), description="Langfuse public API key")
    LANGFUSE_SECRET_KEY: SecretStr = Field(default=SecretStr(""), description="Langfuse secret API key")
    LANGFUSE_HOST: str = Field(
        default="https://cloud.langfuse.com",
        description="Langfuse server URL",
    )

    # ── Trino ─────────────────────────────────────────────────────────────────
    TRINO_HOST: str = "localhost"
    TRINO_PORT: int = 8080
    TRINO_USER: str = "trino"
    TRINO_PASSWORD: SecretStr = SecretStr("")
    TRINO_CATALOG: str = "minio"
    TRINO_SCHEMA: str = "simple_retail"
    TRINO_HTTP_SCHEME: str = "http"
    TRINO_REQUEST_TIMEOUT: float = 30.0
    TRINO_ENABLED: bool = True
    TRINO_VERIFY: bool | str = True
    TRINO_CERT_PATH: str | None = None
    TRINO_KEY_PATH: str | None = None

    # ── Agent (legacy REST chat endpoint) ───────────────────────────────────────
    AGENT_URL: str = "http://localhost:8000"
    AGENT_ENDPOINT: str = "/api/agent/chat"
    AGENT_TIMEOUT: float = 120.0

    # ── MCP Agent (text2sql-onboarding, new connection_id-based interface) ───────
    # Base URL of the agent's MCP server (e.g. http://localhost:8001)
    MCP_AGENT_URL: str = "http://localhost:8001"
    # Numeric connection ID passed to chat_with_agent (matches JEEN_METADATA_CONNECTION_ID)
    MCP_AGENT_CONNECTION_ID: int = 0

    # SQL dialect the agent writes (it executes on Trino). Generated SQL that
    # fails on a connection with another dialect is retried transpiled.
    AGENT_SQL_DIALECT: str = "trino"

    # ── Jeen metadata (connections: SQL execution, schema, dialect) ──────────
    # Golden SQL is written for the Jeen connection's own database, so both
    # expected and generated SQL run through Jeen's run_query by connection_id.
    # Leave the key empty to execute on Trino directly instead.
    JEEN_METADATA_MCP_URL: str = "http://schema-modeler.dev161.internal/api/mcp"
    JEEN_METADATA_MCP_KEY: SecretStr = SecretStr("")
    JEEN_METADATA_TIMEOUT: float = 120.0
    # run_query returns at most this many rows (Jeen caps it at 1000)
    JEEN_RUN_QUERY_MAX_ROWS: int = 1000
    JEEN_MAX_CONCURRENT_QUERIES: int = 4

    # ── Backend (for resolving production tables and query execution) ─────────
    BACKEND_URL: str = "http://localhost:8000"
    BACKEND_QUERY_ENDPOINT: str = "/api/query/execute"
    BACKEND_TIMEOUT: float = 30.0

    # ── Evaluation ────────────────────────────────────────────────────────────
    MAX_CONCURRENT_EVALUATIONS: int = 1
    EVALUATION_HITL_ENABLED: bool = False
    # Budget for executing both SQLs and running all evaluators for one item
    # (the agent call has its own AGENT_TIMEOUT)
    EVALUATION_ITEM_TIMEOUT: float = 600.0
    # Per time-shift offset (two query executions)
    TIME_SHIFT_TIMEOUT: float = 180.0

    # Time shift offsets in days (negative = past)
    TIME_SHIFT_OFFSETS_DAYS: list[int] = [-1, -7, -14, -30, -60]

    # Composite score weights — must sum to 1.0. Result correctness dominates:
    # contains accuracy is the primary metric and time shift guards it against
    # coincidental matches; SQL-text similarity only breaks ties.
    COMPOSITE_WEIGHT_EXECUTION_ACCURACY: float = 0.15
    COMPOSITE_WEIGHT_CONTAINS_ACCURACY: float = 0.40
    COMPOSITE_WEIGHT_SQL_EXACT_MATCH: float = 0.0
    COMPOSITE_WEIGHT_TIME_SHIFT: float = 0.25
    COMPOSITE_WEIGHT_COMPONENT_MATCH: float = 0.10
    COMPOSITE_WEIGHT_SCHEMA_HALLUCINATION: float = 0.05
    COMPOSITE_WEIGHT_DIALECT_ERROR: float = 0.05

    # ── BIRD benchmark (dataset "bird", run on a connection named after a BIRD db_id)
    # Bundled with the service (no internet access needed); relative paths are
    # resolved against the project root.
    BIRD_DATASET_PATH: str = "src/config/bird/bird_verified_cot_2462.jsonl"
    # Append each question's "evidence" hint, as in BIRD's standard setting
    BIRD_INCLUDE_EVIDENCE: bool = True

    # Numeric tolerance for result comparison (decimal places)
    NUMERIC_COMPARISON_TOLERANCE: int = 6

    # ── Service ───────────────────────────────────────────────────────────────
    PORT: int = 5002
    APP_ENV: str = "development"
    LOG_LEVEL: str = "INFO"
    CORS_ORIGINS: list[str] = Field(default=["*"], description="Allowed CORS origins")

    @property
    def agent_chat_url(self) -> str:
        """Full agent chat endpoint URL."""
        return f"{self.AGENT_URL}{self.AGENT_ENDPOINT}"

    @property
    def is_langfuse_configured(self) -> bool:
        """True when Langfuse credentials are present."""
        return bool(
            self.LANGFUSE_PUBLIC_KEY.get_secret_value()
            and self.LANGFUSE_SECRET_KEY.get_secret_value()
        )

    @model_validator(mode="after")
    def _validate_trino_security(self) -> Settings:
        if self.TRINO_PASSWORD and self.TRINO_PASSWORD.get_secret_value():
            if self.TRINO_HTTP_SCHEME.lower() != "https":
                raise ValueError(
                    "TRINO_HTTP_SCHEME must be 'https' when credentials (TRINO_PASSWORD) are configured. "
                    "BasicAuthentication over plaintext HTTP is rejected."
                )
        return self


settings = Settings()

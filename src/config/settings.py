"""
settings.py — Application configuration via pydantic-settings.

All values are read from environment variables or a .env file.
Supports Trino mTLS, basic auth, and no-auth modes.
"""

from __future__ import annotations

import os

from pydantic import Field, SecretStr, field_validator, model_validator
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

    # ── Agent's Trino: where the agent's generated SQL is executed ──────────
    # The agent writes and tests its SQL on its own Trino, where column types
    # can differ from the connection's database (Postgres money → varchar).
    # Catalog every connection's SQL runs in on that Trino — the agent's
    # TRINO_CATALOG (e.g. "dataverse"). Empty: only AGENT_TRINO_CATALOGS run there.
    AGENT_TRINO_CATALOG: str = ""
    # Per-connection exceptions to AGENT_TRINO_CATALOG ("6=adventureworks,7=sales")
    AGENT_TRINO_CATALOGS: str = ""
    AGENT_TRINO_HOST: str | None = None
    AGENT_TRINO_PORT: int = 443
    AGENT_TRINO_USER: str = "trino"
    AGENT_TRINO_SCHEMA: str = "public"
    AGENT_TRINO_HTTP_SCHEME: str = "https"
    AGENT_TRINO_CERT_PATH: str | None = None
    AGENT_TRINO_KEY_PATH: str | None = None
    AGENT_TRINO_VERIFY: bool | str = True
    AGENT_TRINO_REQUEST_TIMEOUT: float = 60.0
    # Same cap as Jeen run_query, so both results are truncated alike
    AGENT_TRINO_MAX_ROWS: int = 1000
    # Queries run on the agent's Trino at once (its databases have few connections)
    AGENT_TRINO_MAX_CONCURRENT_QUERIES: int = 2
    # Run the golden SQL on the agent's Trino too (transpiled), so both results
    # come from the same engine; falls back to the connection when it can't run there
    EXPECTED_SQL_ON_AGENT_ENGINE: bool = True

    # ── Agent (legacy REST chat endpoint) ───────────────────────────────────────
    AGENT_URL: str = "http://localhost:8000"
    AGENT_ENDPOINT: str = "/api/agent/chat"
    AGENT_TIMEOUT: float = 120.0

    # ── MCP Agent (text2sql-onboarding, new connection_id-based interface) ───────
    # Base URL of the agent's MCP server (e.g. http://localhost:8001)
    MCP_AGENT_URL: str = "http://localhost:8001"
    # Numeric connection ID passed to chat_with_agent (matches JEEN_METADATA_CONNECTION_ID)
    MCP_AGENT_CONNECTION_ID: int = 0

    # ── Previous agent (the earlier Text-to-SQL agent, chosen per evaluation) ──
    # Its MCP server (Streamable HTTP) and the tool that answers a question
    PREVIOUS_AGENT_MCP_URL: str = "https://mcp.np.agency.idf.cts/mcp"
    PREVIOUS_AGENT_TOOL_NAME: str = "TextToSQL"
    # Comma-separated "databases" argument. The previous agent works by
    # database, not by connection, so every evaluation sends these.
    PREVIOUS_AGENT_DATABASES: str = "home_command"
    # TLS check: true, false, or a CA bundle path (the private network's CA)
    PREVIOUS_AGENT_VERIFY: bool | str = True
    # Optional bearer token, when the server asks for one
    PREVIOUS_AGENT_TOKEN: SecretStr = SecretStr("")

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

    # ── LLM (order judge) ─────────────────────────────────────────────────────
    # Any OpenAI-compatible Chat Completions endpoint. Private network: set
    # LLM_BASE_URL to the AI service and LLM_MODEL to e.g. Qwen/Qwen3.6-27B-FP8,
    # Qwen/Qwen3.8-27B-FP8 or openai/gpt-oss-120b. Outside it, OPENAI_API_KEY
    # alone uses OpenAI with the default model.
    LLM_API_KEY: SecretStr = SecretStr("")
    OPENAI_API_KEY: SecretStr = SecretStr("")
    LLM_BASE_URL: str = "https://api.openai.com/v1"
    LLM_MODEL: str = "gpt-4.1-mini"
    LLM_TIMEOUT: float = 60.0
    # Decide "does the question ask for an order?" when rows match but order
    # differs and the golden SQL has ORDER BY without LIMIT
    ORDER_JUDGE_ENABLED: bool = True
    # Saved decisions per question (stable across runs); relative to project root
    ORDER_JUDGE_CACHE_PATH: str = "data/order_judgments.json"

    # ── LLM judge (second look at questions contains accuracy failed) ─────────
    # Name of the Langfuse text prompt (label "production") the judge uses,
    # uploaded with scripts/upload_judge_prompt.py. Empty: no judge.
    LLM_JUDGE_PROMPT_NAME: str = ""
    # Model for the judge; empty: LLM_MODEL
    LLM_JUDGE_MODEL: str = ""
    # A "false negative" verdict turns the question into a pass only at this
    # confidence (0-100) or above
    LLM_JUDGE_MIN_CONFIDENCE: int = 80
    LLM_JUDGE_TIMEOUT: float = 120.0

    # ── Backend (for resolving production tables and query execution) ─────────
    BACKEND_URL: str = "http://localhost:8000"
    BACKEND_QUERY_ENDPOINT: str = "/api/query/execute"
    BACKEND_TIMEOUT: float = 30.0

    # ── Evaluation ────────────────────────────────────────────────────────────
    # Questions evaluated at the same time. Each one keeps an agent call open
    # for minutes, so one at a time made a 150-question BIRD run take hours;
    # the agent's Trino and Jeen queries have their own, lower limits
    # (AGENT_TRINO_MAX_CONCURRENT_QUERIES, JEEN_MAX_CONCURRENT_QUERIES).
    MAX_CONCURRENT_EVALUATIONS: int = 4
    EVALUATION_HITL_ENABLED: bool = False
    # Budget for executing both SQLs and running all evaluators for one item
    # (the agent call has its own AGENT_TIMEOUT)
    EVALUATION_ITEM_TIMEOUT: float = 600.0
    # Per time-shift offset (two query executions)
    TIME_SHIFT_TIMEOUT: float = 180.0

    # ── LLM rate limits (HTTP 429) ───────────────────────────────────────────
    # Every LLM call (the agent writing the SQL, the LLM judge, the order judge)
    # that the provider rate limits is retried this many times, waiting
    # LLM_RATE_LIMIT_RETRY_WAIT_SECONDS before the first retry and twice as
    # long before each next one (20s, 40s, 80s). After that the question is
    # marked rate limited, and the studio can rerun it.
    LLM_RATE_LIMIT_MAX_RETRIES: int = Field(default=3, ge=0, le=10)
    LLM_RATE_LIMIT_RETRY_WAIT_SECONDS: float = Field(default=20.0, gt=0, le=3600)

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
    COMPOSITE_WEIGHT_SCHEMA_VALIDITY: float = 0.05
    COMPOSITE_WEIGHT_DIALECT_VALIDITY: float = 0.05

    # ── BIRD benchmark (dataset "bird", run on a connection named after a BIRD db_id)
    # Bundled with the service (no internet access needed); relative paths are
    # resolved against the project root.
    BIRD_DATASET_PATH: str = "src/config/bird/bird_verified_cot_2462.jsonl"
    # Append each question's "evidence" hint, as in BIRD's standard setting
    BIRD_INCLUDE_EVIDENCE: bool = True
    # Bundled BIRD databases (<db_id>.sqlite) that load_bird_databases copies
    # into a connection's database, one schema "<prefix><db_id>" each
    BIRD_DATABASES_DIR: str = "src/config/bird/databases"
    BIRD_SCHEMA_PREFIX: str = "bird_"
    # Set when every BIRD database was pushed to dataverse with
    # push_bird_to_oasis (e.g. "agency__bird-public-benchmark"): gold SQL then
    # reads "<schema>".<db_id>__<table> with cleaned column names
    BIRD_DATAVERSE_SCHEMA: str = ""

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
    def llm_api_key(self) -> str:
        """LLM_API_KEY, falling back to OPENAI_API_KEY."""
        return self.LLM_API_KEY.get_secret_value() or self.OPENAI_API_KEY.get_secret_value()

    @property
    def is_langfuse_configured(self) -> bool:
        """True when Langfuse credentials are present."""
        return bool(
            self.LANGFUSE_PUBLIC_KEY.get_secret_value()
            and self.LANGFUSE_SECRET_KEY.get_secret_value()
        )

    @field_validator(
        "TRINO_VERIFY", "AGENT_TRINO_VERIFY", "PREVIOUS_AGENT_VERIFY", mode="before"
    )
    @classmethod
    def _verify_flag(cls, value: object) -> object:
        """
        "true"/"false" (and 1/0, yes/no, on/off) as booleans; anything else is a
        CA bundle path. An env var is always text, and `bool | str` would keep
        "false" as the path "false" ("[Errno 2] No such file or directory").
        """
        if isinstance(value, str):
            word = value.strip().lower()
            if word in ("false", "0", "no", "off"):
                return False
            if word in ("true", "1", "yes", "on"):
                return True
        return value

    def tls_checks(self) -> list[dict[str, object]]:
        """
        How each TLS setting was read, for the startup log: "on" (system CAs),
        "off" (no check) or "ca_file" with whether that file is in the container.
        """
        checks: list[dict[str, object]] = []
        for name in ("TRINO_VERIFY", "AGENT_TRINO_VERIFY", "PREVIOUS_AGENT_VERIFY"):
            value = getattr(self, name)
            if isinstance(value, bool):
                checks.append({"setting": name, "mode": "on" if value else "off"})
            else:
                checks.append(
                    {
                        "setting": name,
                        "mode": "ca_file",
                        "path": value,
                        "exists": os.path.isfile(value),
                    }
                )
        return checks

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

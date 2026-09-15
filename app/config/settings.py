"""Application settings loaded from environment variables.

All configuration is centralised here so that every module can import a single
``settings`` singleton rather than calling ``os.getenv`` directly throughout
the codebase.  Secrets are intentionally excluded from source control via
``.gitignore`` and the ``.env.example`` file documents every required key.
"""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Typed configuration loaded from environment variables / .env file.

    Pydantic-settings validates every value at startup so misconfiguration is
    caught immediately rather than at runtime.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",          # silently ignore unknown env vars
        case_sensitive=False,
    )

    # ── LLM provider ──────────────────────────────────────────────────────────
    model_provider: str = Field(
        default="",
        description="LLM provider: openai | anthropic | google | azure_openai",
    )
    model_name: str = Field(
        default="",
        description="Model name, e.g. gpt-4o or claude-3-5-sonnet",
    )
    api_key: str = Field(
        default="",
        description="API key for the chosen provider — never hard-code.",
    )

    # ── Azure OpenAI (only used when model_provider == 'azure_openai') ────────
    azure_openai_endpoint: str = Field(default="")
    azure_openai_api_version: str = Field(default="")

    # ── Application behaviour ─────────────────────────────────────────────────
    log_level: str = Field(
        default="INFO",
        description="Log verbosity: DEBUG | INFO | WARNING | ERROR",
    )
    output_dir: str = Field(
        default="output",
        description="Directory where pipeline run artefacts are written.",
    )
    max_agent_retries: int = Field(
        default=3,
        ge=0,
        description="Maximum retry attempts before the Orchestrator escalates a failed step.",
    )


# Module-level singleton — import this directly in other modules.
settings = Settings()

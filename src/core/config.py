"""Validated application settings loaded from the project root .env file."""

from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    db_host: str = "localhost"
    db_port: int = Field(default=5432, ge=1, le=65535)
    db_name: str = Field(min_length=1)
    db_user: str = Field(min_length=1)
    db_password: SecretStr = Field(min_length=1)
    db_pool_min: int = Field(default=1, ge=1)
    db_pool_max: int = Field(default=10, ge=1)

    embedding_model: str = "BAAI/bge-m3"
    embedding_dim: int = Field(default=1024, ge=1)
    embedding_batch_size: int = Field(default=16, ge=1)
    top_k: int = Field(default=5, ge=1)
    openai_api_key: SecretStr | None = None
    openai_model: str = "gpt-4.1-mini"
    ollama_model: str = "qwen3.5:4b"
    ollama_host: str = "http://localhost:11434"
    mistral_api_key: SecretStr | None = None
    mistral_model: str = "mistral-small-latest"
    groundedness_model: str = "qwen3.5:4b"
    judge_num_ctx: int = Field(default=32768, gt=0)
    judge_max_tokens: int = Field(default=4096, gt=0)
    judge_seed: int = 42
    llm_timeout_seconds: float = Field(default=120, gt=0)
    log_level: str = "INFO"

    @model_validator(mode="after")
    def validate_pool_size(self) -> "Settings":
        if self.db_pool_max < self.db_pool_min:
            raise ValueError("DB_POOL_MAX must be greater than or equal to DB_POOL_MIN")
        if self.judge_num_ctx <= self.judge_max_tokens + 256:
            raise ValueError("JUDGE_NUM_CTX must exceed JUDGE_MAX_TOKENS + 256")
        return self

    @property
    def db_connect_kwargs(self) -> dict[str, str | int]:
        """Keyword arguments accepted by psycopg2, including the unwrapped password."""
        return {
            "host": self.db_host,
            "port": self.db_port,
            "dbname": self.db_name,
            "user": self.db_user,
            "password": self.db_password.get_secret_value(),
        }


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Read settings once per process, after environment variables are available."""
    return Settings()

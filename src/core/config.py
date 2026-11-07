"""Validated application settings loaded from the project root .env file."""

from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Literal


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
    ollama_model: str = "qwen3.5:4b"
    ollama_host: str = "http://localhost:11434"
    openrouter_api_key: SecretStr | None = None
    openrouter_model: str = "mistralai/mistral-small-24b-instruct-2501"
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    groundedness_model: str = "qwen3.5:4b"
    judge_num_ctx: int = Field(default=32768, gt=0)
    judge_max_tokens: int = Field(default=5120, gt=0)
    judge_seed: int = 42
    llm_timeout_seconds: float = Field(default=120, gt=0)
    log_level: str = "INFO"
    # "full": tính c_retrieval + c_consensus + c_groundedness (gọi judge).
    # "lite": chỉ c_retrieval + c_consensus, không gọi judge — nhanh hơn, ít thông tin hơn.
    confidence_mode: Literal["full", "lite"] = "full"

    # Similarity Subsystem & Duplicate / Incident Detection
    # Cửa sổ thời gian (phút) phát hiện spam dồn dập từ cùng 1 email
    similarity_spam_window_minutes: int = Field(default=5, ge=1)
    # Cửa sổ thời gian (ngày) kiểm tra câu hỏi lặp lại chưa xử lý dứt điểm (repeat-unresolved)
    similarity_repeat_window_days: int = Field(default=7, ge=1)
    # Ngưỡng cosine similarity để coi là cùng 1 vấn đề hỏi lại (>= 0.85 -> Escalate)
    similarity_repeat_threshold: float = Field(default=0.85, ge=0.0, le=1.0)
    # Cửa sổ thời gian trượt (phút) để gom nhóm các ticket sự cố diện rộng (mass incident)
    similarity_incident_window_minutes: int = Field(default=30, ge=1)
    # Ngưỡng cosine similarity giữa các ticket khác người gửi để xem xét cùng 1 sự cố
    similarity_incident_threshold: float = Field(default=0.82, ge=0.0, le=1.0)
    # Số ticket tối thiểu từ các user khác nhau trong cửa sổ để chính thức kích hoạt tạo Incident
    similarity_incident_min_tickets: int = Field(default=5, ge=2)
    # Ngưỡng cosine similarity với ticket đã closed trong quá khứ để gắn cờ lỗi tái phát (recurring)
    similarity_recurring_threshold: float = Field(default=0.88, ge=0.0, le=1.0)

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
        # Ví dụ dict kết nối (giá trị giả): {"host": "localhost", "port": 5432,
        # "dbname": "helpdesk", "user": "app", "password": "<mật khẩu>"}.
        # password đã được giải bọc SecretStr ở đây; không log dict này.
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

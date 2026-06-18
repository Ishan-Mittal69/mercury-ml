from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # ── Model ─────────────────────────────────────────────────────────────────
    MODEL_PATH: str = ""
    CONFIDENCE_THRESHOLD: float = Field(default=0.7, ge=0.0, le=1.0)

    # ── Gemini ────────────────────────────────────────────────────────────────
    GEMINI_API_KEY: str = ""
    GEMINI_MODEL: str = "gemini-2.5-flash"
    GEMINI_TIMEOUT: float = Field(default=60.0, gt=0)
    GEMINI_MAX_RETRIES: int = Field(default=3, ge=1, le=10)

    # ── Request limits ────────────────────────────────────────────────────────
    MAX_SEGMENTS_PER_REQUEST: int = Field(default=500, ge=1)
    MAX_TEXT_LENGTH: int = Field(default=2000, ge=1)

    # ── Server ────────────────────────────────────────────────────────────────
    PORT: int = Field(default=8000, ge=1, le=65535)
    LOG_LEVEL: str = "info"
    LOG_FORMAT: str = "json"  # "json" | "text"

    @field_validator("LOG_LEVEL")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        allowed = {"debug", "info", "warning", "error", "critical"}
        if v.lower() not in allowed:
            raise ValueError(f"LOG_LEVEL must be one of {allowed}")
        return v.lower()

    @property
    def gemini_ready(self) -> bool:
        return bool(self.GEMINI_API_KEY)

    @property
    def model_configured(self) -> bool:
        return bool(self.MODEL_PATH)


settings = Settings()

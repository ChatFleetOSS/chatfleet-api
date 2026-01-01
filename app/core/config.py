# generated-by: codex-agent 2025-02-15T00:12:00Z
"""
Runtime configuration loading for the ChatFleet backend.

We read defaults from `.env.example` (per safety policy) and allow environment
overrides for local development and CI.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from dotenv import dotenv_values
from pydantic import BaseModel, Field, ConfigDict, field_validator


class Settings(BaseModel):
    """Typed settings derived from .env.example with environment overrides."""

    model_config = ConfigDict(populate_by_name=True, validate_assignment=True)

    mongo_uri: str = Field(alias="MONGO_URI")
    auto_start_mongo: bool = Field(alias="AUTO_START_MONGO")
    mongo_bin: str = Field(alias="MONGO_BIN")
    mongo_config_path: Optional[Path] = Field(default=None, alias="MONGO_CONFIG_PATH")
    mongo_db_path: Path = Field(alias="MONGO_DB_PATH")
    mongo_startup_timeout_s: int = Field(alias="MONGO_STARTUP_TIMEOUT_S")
    jwt_secret: str = Field(alias="JWT_SECRET")
    chat_model: str = Field(alias="CHAT_MODEL")
    embed_model: str = Field(alias="EMBED_MODEL")
    index_dir: Path = Field(alias="INDEX_DIR")
    upload_dir: Path = Field(alias="UPLOAD_DIR")
    max_upload_mb: int = Field(alias="MAX_UPLOAD_MB")
    sse_heartbeat_ms: int = Field(alias="SSE_HEARTBEAT_MS")
    top_k_default: int = Field(alias="TOP_K_DEFAULT")
    temperature_default: float = Field(alias="TEMPERATURE_DEFAULT")
    cors_origins: List[str] = Field(alias="CORS_ORIGINS")

    @field_validator("cors_origins", mode="before")
    @classmethod
    def split_origins(cls, value: str | List[str]) -> List[str]:
        if isinstance(value, list):
            return value
        return [origin.strip() for origin in value.split(",") if origin.strip()]

    @field_validator("index_dir", "upload_dir", mode="before")
    @classmethod
    def ensure_path(cls, value: str | Path) -> Path:
        return Path(value)

    @field_validator("auto_start_mongo", mode="before")
    @classmethod
    def parse_bool(cls, value: bool | int | str) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, int):
            return value != 0
        normalized = value.strip().lower()
        return normalized in {"1", "true", "yes", "on"}

    @field_validator("mongo_config_path", mode="before")
    @classmethod
    def empty_string_to_none(cls, value: Optional[str | Path]) -> Optional[Path]:
        if value in (None, ""):
            return None
        return Path(value)


def load_settings() -> Settings:
    """Load configuration using `.env.example` as the safety baseline."""

    repo_root = Path(__file__).resolve().parents[2]
    env_example = repo_root / ".env.example"
    defaults = dotenv_values(env_example) if env_example.exists() else {}

    # Environment variables override the example defaults.
    import os

    merged: dict[str, Optional[str]] = {**defaults, **dict(os.environ)}

    local_storage = repo_root / "var"
    fallback_index = local_storage / "faiss"
    fallback_uploads = local_storage / "uploads"
    fallback_mongo = local_storage / "mongo"

    required = {
        "MONGO_URI": "mongodb://localhost:27017/chatfleet",
        "AUTO_START_MONGO": "0",
        "MONGO_BIN": "mongod",
        "MONGO_CONFIG_PATH": "",
        "MONGO_DB_PATH": str(fallback_mongo),
        "MONGO_STARTUP_TIMEOUT_S": "20",
        "JWT_SECRET": "changeme",
        "CHAT_MODEL": "gpt-4o-mini",
        "EMBED_MODEL": "BAAI/bge-m3",
        "INDEX_DIR": "/var/lib/chatfleet/faiss",
        "UPLOAD_DIR": "/var/lib/chatfleet/uploads",
        "MAX_UPLOAD_MB": "50",
        "SSE_HEARTBEAT_MS": "15000",
        "TOP_K_DEFAULT": "12",
        "TEMPERATURE_DEFAULT": "0.2",
        "CORS_ORIGINS": "http://localhost:3000",
    }
    for key, fallback in required.items():
        merged.setdefault(key, fallback)

    settings = Settings(**merged)  # type: ignore[arg-type]

    def _ensure_directory(path: Path, fallback: Path) -> Path:
        try:
            path.mkdir(parents=True, exist_ok=True)
            return path
        except PermissionError:
            fallback.mkdir(parents=True, exist_ok=True)
            return fallback

    settings.index_dir = _ensure_directory(settings.index_dir, fallback_index)
    settings.upload_dir = _ensure_directory(settings.upload_dir, fallback_uploads)
    settings.mongo_db_path = _ensure_directory(settings.mongo_db_path, fallback_mongo)

    # Security: require a strong JWT secret in all environments
    secret = settings.jwt_secret or ""
    if secret == "changeme" or len(secret) < 32:
        raise RuntimeError(
            "JWT_SECRET is weak or unset. Set a random 32+ char secret via environment."
        )
    return settings


settings = load_settings()

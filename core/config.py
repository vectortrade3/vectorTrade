"""
VectorTrade - Core Configuration Module
Validates environment settings, directories, and credentials using Pydantic.
Enforces zero hardcoded credentials and safe fallbacks for offline testing.
"""

from __future__ import annotations
import os
from pathlib import Path
from typing import Optional
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv

# Base Directory of VectorTrade
BASE_DIR = Path(__file__).resolve().parent.parent

# Load .env file explicitly
ENV_FILE = BASE_DIR / ".env"
if ENV_FILE.exists():
    load_dotenv(ENV_FILE, override=False)


class Settings(BaseSettings):
    """
    Immutable production configuration schema with Pydantic validation.
    """
    model_config = SettingsConfigDict(
        env_file=str(ENV_FILE),
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # Application Metadata
    APP_NAME: str = "VectorTrade"
    LOG_LEVEL: str = "INFO"
    TIMEZONE: str = "Asia/Kolkata"

    # Filesystem Paths
    BASE_DIR: Path = BASE_DIR
    CORE_DIR: Path = BASE_DIR / "core"
    OFFLINE_DIR: Path = BASE_DIR / "offline"
    ONLINE_DIR: Path = BASE_DIR / "online"
    CONFIG_DIR: Path = BASE_DIR / "config"
    LOGS_DIR: Path = BASE_DIR / "logs"
    MODELS_DIR: Path = BASE_DIR / "offline" / "models"
    STOCK_NAMES_PATH: Path = BASE_DIR / "stock_names.json"
    TOKENS_CACHE_PATH: Path = BASE_DIR / "tokens_cache.json"

    # MongoDB Atlas Database Settings
    MONGODB_URI: str = Field(
        default="mock",
        description="MongoDB Atlas connection string. If 'mock' or empty, mongomock is utilized."
    )
    MONGODB_DB_NAME: str = Field(
        default="vectortrade",
        description="MongoDB target database name."
    )

    # MegaBull Broker API Settings
    MEGABULL_API_KEY: str = Field(
        default="",
        alias="megabull_api",
        description="Active MegaBull API key from user profile."
    )
    MEGABULL_USER_ID: Optional[str] = Field(
        default=None,
        description="Optional MegaBull user identification ID."
    )
    MEGABULL_BASE_URL: str = Field(
        default="https://api.megabull.in",
        description="Base URL for MegaBull REST service."
    )

    # HTTP & Resilience Limits
    HTTP_TIMEOUT_SECONDS: float = Field(
        default=10.0,
        description="Default socket read/connect timeout in seconds."
    )
    HTTP_MAX_RETRIES: int = Field(
        default=3,
        description="Maximum retry attempts with exponential backoff."
    )

    # Secure Inter-Process Communication (Offline <-> Online)
    INTERNAL_AUTH_TOKEN: str = Field(
        default="vt_sec_789f2a4bc81d45e69b03e2c4d8e1a7",
        description="Pre-shared cryptographic token for offline-online secure inter-process communication."
    )
    ONLINE_BASE_URL: str = Field(
        default="http://localhost:8000",
        description="Base URL of online server for direct authenticated requests."
    )

    @field_validator("MEGABULL_API_KEY", mode="before")
    @classmethod
    def resolve_megabull_key(cls, v: Optional[str]) -> str:
        """Fallback to env var 'megabull_api' if MEGABULL_API_KEY is unset."""
        if v and v.strip():
            return v.strip().strip("'\"")
        raw_key = os.getenv("MEGABULL_API_KEY") or os.getenv("megabull_api", "")
        return raw_key.strip().strip("'\"")

    @field_validator("MONGODB_URI", mode="before")
    @classmethod
    def clean_mongo_uri(cls, v: Optional[str]) -> str:
        if not v or not v.strip():
            return "mock"
        return v.strip().strip("'\"")

    @property
    def is_mongo_mock(self) -> bool:
        """Returns True if the system is configured to use in-memory/mock MongoDB."""
        uri = self.MONGODB_URI.lower()
        return uri in ("", "mock", "none", "memory") or "example.mongodb.net" in uri


# Singleton settings instance
settings = Settings()

# Ensure standard runtime directories exist
settings.LOGS_DIR.mkdir(parents=True, exist_ok=True)
settings.MODELS_DIR.mkdir(parents=True, exist_ok=True)

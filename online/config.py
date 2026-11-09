"""
VectorTrade - Online Application Configuration
Loads configuration from environment variables or .env file.
"""

import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = PROJECT_ROOT / ".env"
if ENV_PATH.exists():
    load_dotenv(ENV_PATH)


class Settings(BaseSettings):
    """Online application settings."""
    model_config = SettingsConfigDict(env_file=str(ENV_PATH), extra="allow")

    app_name: str = os.getenv("APP_NAME", "VectorTrade Online")
    env: str = os.getenv("ENVIRONMENT", "production")
    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = int(os.getenv("PORT", "8080"))
    
    # MongoDB Atlas settings
    mongodb_uri: str = os.getenv("MONGODB_URI", "")
    mongodb_database: str = os.getenv("MONGODB_DATABASE", "vectortrade")
    mongo_timeout_ms: int = int(os.getenv("MONGO_TIMEOUT_MS", "5000"))
    
    # Trading Defaults
    timezone: str = os.getenv("TIMEZONE", "Asia/Kolkata")
    default_risk_budget: float = float(os.getenv("DEFAULT_RISK_BUDGET", "10000.0"))

    # Authentication & Session Security
    auth_username: str = os.getenv("AUTH_USERNAME", "root")
    auth_password: str = os.getenv("AUTH_PASSWORD", "root")
    session_secret_key: str = os.getenv("SESSION_SECRET_KEY", "vectortrade-cloud-session-secret-2026-secure")
    session_max_age_seconds: int = int(os.getenv("SESSION_MAX_AGE_SECONDS", "604800"))  # 7 days


settings = Settings()

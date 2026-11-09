"""
VectorTrade - Online Application Configuration
Loads configuration from environment variables or .env file.
"""

import os
from pathlib import Path
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = PROJECT_ROOT / ".env"
if ENV_PATH.exists():
    load_dotenv(ENV_PATH)


class Settings(BaseSettings):
    """Online application settings."""
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

    class Config:
        env_file = str(ENV_PATH)
        extra = "allow"


settings = Settings()

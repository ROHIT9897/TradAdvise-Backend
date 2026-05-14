from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    REDIS_URL: str
    NEWS_API_KEY: str
    ALPHA_VANTAGE_KEY: str = ""
    SENTRY_DSN: Optional[str] = None
    SECRET_KEY: str = "default_secret_key"

    # Cache TTLs (seconds)
    LIVE_PRICE_TTL: int = 60
    PREDICTION_TTL: int = 1800    # 30 minutes
    BACKTEST_TTL: int = 21600     # 6 hours
    NEWS_TTL: int = 3600          # 1 hour
    HISTORICAL_TTL: int = 86400   # 24 hours

    # FinBERT toggle for low memory servers
    DISABLE_FINBERT: str = "false"

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": True,
        "extra": "ignore"
    }


settings = Settings()
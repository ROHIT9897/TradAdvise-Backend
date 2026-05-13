
import logging
import sentry_sdk
from fastapi import FastAPI
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from api.routes import router, limiter
from config import settings

# Sentry crash monitoring
if settings.SENTRY_DSN and settings.SENTRY_DSN.startswith("https://"):
    sentry_sdk.init(dsn=settings.SENTRY_DSN, traces_sample_rate=0.1)

logging.basicConfig(level=logging.INFO)

app = FastAPI(title="StockAI API", version="1.0.0")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.include_router(router, prefix="/api/v1")

@app.on_event("startup")
async def startup():
    logging.info("StockAI API started")
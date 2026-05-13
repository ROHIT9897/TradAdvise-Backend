import redis.asyncio as aioredis
import json
import logging
from typing import Optional, Any
from config import settings

logger = logging.getLogger(__name__)

class RedisCache:
    def __init__(self):
        self.client = None
        self._connect()

    def _connect(self):
        try:
            self.client = aioredis.from_url(
                settings.REDIS_URL,
                decode_responses=True,
                socket_connect_timeout=2,    # fail fast — don't hang
                socket_timeout=2,
            )
        except Exception as e:
            logger.warning(f"Redis init failed: {e}")
            self.client = None

    async def get(self, key: str) -> Optional[Any]:
        if self.client is None:
            return None                      # no cache — continue without it
        try:
            value = await self.client.get(key)
            return json.loads(value) if value else None
        except Exception as e:
            logger.warning(f"Redis GET failed for {key}: {e}")
            return None                      # fail silently — never crash

    async def set(self, key: str, value: Any, ttl: int):
        if self.client is None:
            return                           # skip cache — continue without it
        try:
            await self.client.setex(
                key,
                ttl,
                json.dumps(value, default=str)
            )
        except Exception as e:
            logger.warning(f"Redis SET failed for {key}: {e}")
            # Never crash — cache failure is not fatal

    async def delete(self, key: str):
        if self.client is None:
            return
        try:
            await self.client.delete(key)
        except Exception as e:
            logger.warning(f"Redis DELETE failed: {e}")

    async def exists(self, key: str) -> bool:
        if self.client is None:
            return False
        try:
            return await self.client.exists(key) > 0
        except:
            return False

    async def ping(self) -> bool:
        if self.client is None:
            return False
        try:
            return await self.client.ping()
        except:
            return False

cache = RedisCache()
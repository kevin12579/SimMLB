import asyncio
from abc import ABC, abstractmethod

import aiohttp
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from src.common.exceptions import RateLimitError, DataCollectionError
from src.common.logger import get_logger

logger = get_logger(__name__)

# 동시 HTTP 요청 3개로 제한
_semaphore = asyncio.Semaphore(3)


class BaseCollector(ABC):
    BASE_URL: str = ""

    def __init__(self) -> None:
        self._session: aiohttp.ClientSession | None = None

    async def __aenter__(self) -> "BaseCollector":
        self._session = aiohttp.ClientSession(
            headers={"User-Agent": "MLB-Prediction-Bot/1.0"},
            timeout=aiohttp.ClientTimeout(total=30),
        )
        return self

    async def __aexit__(self, *args: object) -> None:
        if self._session:
            await self._session.close()

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=1, max=30),
        retry=retry_if_exception_type(RateLimitError),
        reraise=True,
    )
    async def _get(
        self,
        url: str,
        params: dict | None = None,
        timeout: float | None = None,
    ) -> dict | list:
        assert self._session is not None, "Use async context manager"
        kwargs: dict = {"params": params}
        if timeout is not None:
            kwargs["timeout"] = aiohttp.ClientTimeout(total=timeout)
        async with _semaphore:
            async with self._session.get(url, **kwargs) as resp:
                if resp.status == 429:
                    retry_after = int(resp.headers.get("Retry-After", 60))
                    logger.warning("Rate limit hit on %s, waiting %ds", url, retry_after)
                    await asyncio.sleep(retry_after)
                    raise RateLimitError(f"429 from {url}")
                if resp.status >= 500:
                    raise DataCollectionError(f"Server error {resp.status} from {url}")
                resp.raise_for_status()
                return await resp.json()

import asyncio
from functools import wraps
from typing import Callable, TypeVar

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from src.common.exceptions import RateLimitError

T = TypeVar("T")

# 최대 5회 재시도, 지수 백오프 (1s → 2s → 4s → 8s → 16s)
retry_on_rate_limit = retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=1, max=30),
    retry=retry_if_exception_type(RateLimitError),
    reraise=True,
)

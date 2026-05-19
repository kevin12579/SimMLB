"""LLM 추상화 레이어 — OpenAI / Groq 전환 가능"""
from __future__ import annotations
from abc import ABC, abstractmethod

from config.settings import settings
from src.common.logger import get_logger

logger = get_logger(__name__)


class LLMClient(ABC):
    @abstractmethod
    async def complete(self, prompt: str) -> str: ...


class OpenAIClient(LLMClient):
    def __init__(self) -> None:
        from openai import AsyncOpenAI
        self._client = AsyncOpenAI(api_key=settings.openai_api_key)
        self._model  = settings.openai_model

    async def complete(self, prompt: str) -> str:
        resp = await self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.3,
        )
        return resp.choices[0].message.content or ""


class GroqClient(LLMClient):
    def __init__(self) -> None:
        from openai import AsyncOpenAI  # Groq는 OpenAI 호환 API 사용
        self._client = AsyncOpenAI(
            api_key=settings.groq_api_key,
            base_url="https://api.groq.com/openai/v1",
        )
        self._model = settings.groq_model

    async def complete(self, prompt: str) -> str:
        resp = await self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.3,
        )
        return resp.choices[0].message.content or ""


def get_llm_client() -> LLMClient:
    provider = settings.llm_provider.lower()
    if provider == "groq":
        logger.info("Using Groq LLM (%s)", settings.groq_model)
        return GroqClient()
    logger.info("Using OpenAI LLM (%s)", settings.openai_model)
    return OpenAIClient()

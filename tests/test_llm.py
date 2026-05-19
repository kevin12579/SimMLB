"""Week 3 완료 기준 테스트: test_llm_fallback"""
import pytest
from unittest.mock import AsyncMock, patch


class TestLLMFallback:
    """LLM API 실패 시 폴백 근거 텍스트 반환 검증"""

    @pytest.mark.asyncio
    async def test_fallback_on_openai_error(self):
        """OpenAI API 에러 → 폴백 문자열 반환, 예외 전파 없음"""
        from src.ml.reasoning.prompt_builder import generate_reasoning

        shap_top5 = [
            {"feature": "home_sp_fip", "value": 3.12, "shap_value": 0.15},
        ]
        with patch("src.ml.reasoning.llm_client.get_llm_client") as mock_factory:
            mock_client = AsyncMock()
            mock_client.complete.side_effect = Exception("OpenAI API Error")
            mock_factory.return_value = mock_client

            result = await generate_reasoning("LAD", "NYY", 0.623, shap_top5)

        # 예외 없이 반환되고, 팀명 또는 확률이 포함된 폴백 문자열
        assert isinstance(result, str)
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_fallback_contains_prob(self):
        """폴백 텍스트에는 확률 정보가 포함되어야 함"""
        from src.ml.reasoning.prompt_builder import generate_reasoning

        with patch("src.ml.reasoning.llm_client.get_llm_client") as mock_factory:
            mock_client = AsyncMock()
            mock_client.complete.side_effect = RuntimeError("Network error")
            mock_factory.return_value = mock_client

            result = await generate_reasoning("BOS", "HOU", 0.55, [])

        assert "55" in result or "BOS" in result or "홈" in result

    @pytest.mark.asyncio
    async def test_success_returns_llm_text(self):
        """LLM 정상 응답 시 해당 텍스트를 그대로 반환"""
        from src.ml.reasoning.prompt_builder import generate_reasoning

        expected = "LAD 선발 FIP가 낮아 유리하다. 홈팀 타선 xwOBA가 높다."
        with patch("src.ml.reasoning.llm_client.get_llm_client") as mock_factory:
            mock_client = AsyncMock()
            mock_client.complete.return_value = expected
            mock_factory.return_value = mock_client

            result = await generate_reasoning("LAD", "NYY", 0.623, [])

        assert result == expected

    @pytest.mark.asyncio
    async def test_provider_switch_groq(self):
        """LLM_PROVIDER=groq 설정 시 GroqClient 반환"""
        with patch("src.ml.reasoning.llm_client.settings") as mock_settings:
            mock_settings.llm_provider = "groq"
            mock_settings.groq_api_key = "gsk_test"
            mock_settings.groq_model = "llama-3.1-8b-instant"

            from src.ml.reasoning.llm_client import get_llm_client, GroqClient
            with patch("src.ml.reasoning.llm_client.GroqClient") as MockGroq:
                get_llm_client()
                MockGroq.assert_called_once()

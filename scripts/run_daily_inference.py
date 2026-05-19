"""GitHub Actions Cron 또는 수동 실행용 추론 스크립트"""
import asyncio
import sys
from datetime import date

sys.path.insert(0, ".")

from src.db.session import get_session
from src.ml.prediction_service import run_daily_predictions
from src.common.logger import get_logger

logger = get_logger("daily_inference")


async def main() -> None:
    today = date.today()
    logger.info("=== 일일 추론 시작: %s ===", today)
    with get_session() as session:
        count = await run_daily_predictions(today, session)
    logger.info("=== 일일 추론 완료: %d경기 예측 ===", count)


if __name__ == "__main__":
    asyncio.run(main())

"""GitHub Actions Cron 또는 수동 실행용 추론 스크립트 (BBref 피처 기반)"""
import sys
sys.path.insert(0, ".")

from scripts.run_inference_v2 import run
import asyncio

if __name__ == "__main__":
    asyncio.run(run())

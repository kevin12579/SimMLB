"""GitHub Actions Cron / 수동 실행용 추론 — v2 모델 + Live Feed 통합.

Render Worker 다운 시 폴백. 모델 v2 PKL이 없으면 v1로 자동 폴백.
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, ".")


def _has_v2_models() -> bool:
    md = Path("models")
    return (md / "lgbm_v2.pkl").exists() and (md / "calibrator_v2.pkl").exists()


if __name__ == "__main__":
    if _has_v2_models():
        from scripts.run_inference_v3 import run_all_today as run
    else:
        # v2 학습 전이면 v1로 폴백
        from scripts.run_inference_v2 import run
    asyncio.run(run())

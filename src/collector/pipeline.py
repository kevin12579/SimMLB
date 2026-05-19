"""APScheduler 기반 일일 자동 파이프라인 (KST 기준)"""
import asyncio
from datetime import date, timedelta

import aiohttp
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from src.collector.mlb_statsapi_client import MLBStatsAPIClient
from src.collector.statcast_collector import fetch_statcast_range, save_statcast
from src.collector.fangraphs_collector import (
    fetch_pitching_stats, fetch_batting_stats,
    save_pitching_stats, save_batting_stats,
)
from src.collector.weather_client import WeatherClient
from src.db.session import get_session
from src.common.logger import get_logger
from config.settings import settings

logger = get_logger(__name__)


# ──────────────────────────────────────────
# 개별 파이프라인 태스크
# ──────────────────────────────────────────

def run_morning_pipeline() -> None:
    """07:00 KST — 전일 MLB 결과 수집 + 팀 스냅샷 업데이트"""
    async def _inner() -> None:
        yesterday = date.today() - timedelta(days=1)
        async with MLBStatsAPIClient() as client:
            with get_session() as session:
                await client.update_game_results(yesterday, session)
                logger.info("[morning] Game results updated for %s", yesterday)
        _notify_discord(f"✅ 07:00 파이프라인 완료 — {yesterday} 결과 업데이트")

    _run(run_morning_pipeline.__name__, _inner)


def run_player_stats_update() -> None:
    """07:30 KST — 선수 시즌 통계 갱신 (FanGraphs)"""
    async def _inner() -> None:
        season = date.today().year
        as_of = date.today()
        with get_session() as session:
            pitch_df = await fetch_pitching_stats(season)
            await save_pitching_stats(pitch_df, season, as_of, session)
            bat_df = await fetch_batting_stats(season)
            await save_batting_stats(bat_df, season, as_of, session)
        _notify_discord(f"✅ 07:30 파이프라인 완료 — {season} 시즌 선수 통계 갱신")

    _run(run_player_stats_update.__name__, _inner)


def run_statcast_pipeline() -> None:
    """12:00 KST — 전일 Statcast 수집"""
    async def _inner() -> None:
        yesterday = date.today() - timedelta(days=1)
        with get_session() as session:
            df = await fetch_statcast_range(yesterday, yesterday)
            count = await save_statcast(df, session)
        _notify_discord(f"✅ 12:00 파이프라인 완료 — Statcast {count}개 저장")

    _run(run_statcast_pipeline.__name__, _inner)


def run_fangraphs_update() -> None:
    """12:30 KST — FanGraphs 리더보드 재수집"""
    run_player_stats_update()


def run_weather_collection() -> None:
    """18:00 KST — 당일 경기 날씨 수집"""
    async def _inner() -> None:
        today = date.today()
        async with WeatherClient() as client:
            with get_session() as session:
                await client.sync_weather_for_date(today, session)
        _notify_discord(f"✅ 18:00 파이프라인 완료 — {today} 날씨 수집")

    _run(run_weather_collection.__name__, _inner)


def sync_lineups_task() -> None:
    """18:30 KST — 당일 라인업 확정 수집"""
    async def _inner() -> None:
        today = date.today()
        async with MLBStatsAPIClient() as client:
            with get_session() as session:
                await client.sync_schedule(today, session)
        _notify_discord(f"✅ 18:30 파이프라인 완료 — {today} 라인업 동기화")

    _run(sync_lineups_task.__name__, _inner)


def run_inference_pipeline() -> None:
    """19:30 KST — ML 추론 + LLM 근거 생성 → DB 저장"""
    async def _inner() -> None:
        from scripts.run_inference_v2 import run
        today = date.today()
        await run(today)
        _notify_discord(f"✅ 19:30 추론 완료 — {today} 경기 예측 생성")

    _run(run_inference_pipeline.__name__, _inner)


def retrain_models_task() -> None:
    """일요일 03:00 KST — 주간 모델 재학습"""
    import subprocess, sys
    logger.info("[retrain] 주간 모델 재학습 시작")
    try:
        result = subprocess.run(
            [sys.executable, "scripts/build_and_train.py"],
            capture_output=True, text=True, timeout=3600,
        )
        if result.returncode == 0:
            logger.info("[retrain] 재학습 완료")
            _notify_discord("✅ 주간 모델 재학습 완료")
        else:
            logger.error("[retrain] 재학습 실패:\n%s", result.stderr[-2000:])
            _notify_discord(f"❌ 주간 모델 재학습 실패: {result.stderr[-200:]}")
    except subprocess.TimeoutExpired:
        logger.error("[retrain] 재학습 1시간 초과")
        _notify_discord("❌ 주간 모델 재학습 타임아웃")


# ──────────────────────────────────────────
# 헬퍼
# ──────────────────────────────────────────

def _run(name: str, coro_factory: object) -> None:
    """동기 래퍼: 새 이벤트 루프에서 코루틴 실행"""
    try:
        logger.info("[pipeline] Starting %s", name)
        asyncio.run(coro_factory())  # type: ignore[arg-type]
    except Exception as e:
        logger.error("[pipeline] %s FAILED: %s", name, e, exc_info=True)
        _notify_discord(f"❌ {name} 실패: {e}")


def _notify_discord(message: str) -> None:
    if not settings.discord_webhook_url:
        return
    try:
        import urllib.request, json
        payload = json.dumps({"content": message}).encode()
        req = urllib.request.Request(
            settings.discord_webhook_url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        urllib.request.urlopen(req, timeout=10)
    except Exception as e:
        logger.warning("Discord notify failed: %s", e)


# ──────────────────────────────────────────
# 스케줄러 설정 (KST = UTC+9)
# ──────────────────────────────────────────

def setup_scheduler() -> BackgroundScheduler:
    scheduler = BackgroundScheduler(timezone="Asia/Seoul")

    scheduler.add_job(run_morning_pipeline,     CronTrigger(hour=7,  minute=0))
    scheduler.add_job(run_player_stats_update,  CronTrigger(hour=7,  minute=30))
    scheduler.add_job(run_statcast_pipeline,    CronTrigger(hour=12, minute=0))
    scheduler.add_job(run_fangraphs_update,     CronTrigger(hour=12, minute=30))
    scheduler.add_job(run_weather_collection,   CronTrigger(hour=18, minute=0))
    scheduler.add_job(sync_lineups_task,        CronTrigger(hour=18, minute=30))
    scheduler.add_job(run_inference_pipeline,   CronTrigger(hour=19, minute=30))
    scheduler.add_job(retrain_models_task,      CronTrigger(day_of_week="sun", hour=3, minute=0))

    return scheduler


if __name__ == "__main__":
    scheduler = setup_scheduler()
    scheduler.start()
    logger.info("Scheduler started. Press Ctrl+C to exit.")
    try:
        import time
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        scheduler.shutdown()

"""APScheduler 기반 일일 자동 파이프라인 — 옵션 B + 라이브 폴러 + 포스트게임 통합.

타임라인 (KST):
  03:00 (일) 주간 모델 재학습
  06:30 BBref 스텔스 스크래퍼 (pitching/batting overwrite)
  07:00 전일 결과 + 오늘 일정 사전 동기화 (옵션 B)
  07:30 FanGraphs 선수 통계
  12:00 전일 Statcast (1일 delta)
  12:30 FanGraphs 리더보드 재수집
  13:00 마스터: 오늘 경기마다 T-120/T-15/T+0 동적 등록
  T-120  pre_game_sync (Live Feed 1차)
  T-15   dynamic_inference (47피처 + LLM)
  T+0    live_poller 1분 polling (Final 감지 시 자가 종료 + postgame 트리거)
  19:30  fallback (동적 워커 누락 시 일괄 보충)
"""
from __future__ import annotations

import asyncio
from datetime import date, datetime, timedelta, timezone

try:
    from zoneinfo import ZoneInfo
except ImportError:
    import pytz as ZoneInfo  # type: ignore

import aiohttp
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.date import DateTrigger
from apscheduler.triggers.interval import IntervalTrigger

from config.settings import settings
from src.collector.fangraphs_collector import (
    fetch_batting_stats,
    fetch_pitching_stats,
    save_batting_stats,
    save_pitching_stats,
)
from src.collector.mlb_statsapi_client import MLBStatsAPIClient
from src.collector.statcast_collector import fetch_statcast_range, save_statcast
from src.common.logger import get_logger
from src.db.session import get_session

logger = get_logger(__name__)


# ──────────────────────────────────────────
# 1. 고정 스케줄 태스크
# ──────────────────────────────────────────

def run_morning_pipeline() -> None:
    """07:00 KST — 전일 결과 + 오늘 일정 사전 동기화 (옵션 B)"""
    async def _inner() -> None:
        yesterday = date.today() - timedelta(days=1)
        today = date.today()
        async with MLBStatsAPIClient() as client:
            with get_session() as session:
                await client.update_game_results(yesterday, session)
                await client.sync_schedule(today, session)  # 옵션 B
        _notify_discord(
            f"✅ 07:00 — {yesterday} 결과 + {today} 일정 사전 동기화"
        )
    _run(run_morning_pipeline.__name__, _inner)


def run_bref_daily_update() -> None:
    """06:30 KST — BBref pitching/batting 시즌 누적 덮어쓰기"""
    def _inner() -> None:
        from src.collector.bref_scraper import update_bref_season
        season = date.today().year
        counts = update_bref_season(season)
        _notify_discord(
            f"✅ 06:30 BBref — pitching:{counts.get('pitching', 0)} "
            f"batting:{counts.get('batting', 0)}"
        )
    try:
        logger.info("[pipeline] Starting run_bref_daily_update")
        _inner()
    except Exception as e:
        logger.error("[pipeline] run_bref_daily_update FAILED: %s", e, exc_info=True)
        _notify_discord(f"❌ BBref 업데이트 실패: {e}")


def run_player_stats_update() -> None:
    """07:30 KST — FanGraphs 선수 통계 갱신"""
    async def _inner() -> None:
        season = date.today().year
        as_of = date.today()
        with get_session() as session:
            pitch_df = await fetch_pitching_stats(season)
            await save_pitching_stats(pitch_df, season, as_of, session)
            bat_df = await fetch_batting_stats(season)
            await save_batting_stats(bat_df, season, as_of, session)
        _notify_discord(f"✅ 07:30 — {season} 시즌 선수 통계 갱신")
    _run(run_player_stats_update.__name__, _inner)


def run_statcast_pipeline() -> None:
    """12:00 KST — 전일 Statcast 1일 delta append"""
    async def _inner() -> None:
        yesterday = date.today() - timedelta(days=1)
        with get_session() as session:
            df = await fetch_statcast_range(yesterday, yesterday)
            count = await save_statcast(df, session)
        _notify_discord(f"✅ 12:00 Statcast — {count}개 저장")
    _run(run_statcast_pipeline.__name__, _inner)


def run_fangraphs_update() -> None:
    """12:30 KST — FanGraphs 리더보드 재수집"""
    run_player_stats_update()


def run_inference_fallback() -> None:
    """19:30 KST 폴백 — 동적 워커 누락 경기 일괄 보충"""
    async def _inner() -> None:
        from scripts.run_inference_v3 import run_all_today
        await run_all_today()
        _notify_discord("✅ 19:30 폴백 추론 완료")
    _run("inference_fallback", _inner)


def retrain_models_task() -> None:
    """일요일 03:00 KST — 주간 모델 재학습"""
    import subprocess
    import sys
    logger.info("[retrain] 주간 모델 재학습 시작")
    try:
        result = subprocess.run(
            [sys.executable, "scripts/build_and_train_v2.py"],
            capture_output=True, text=True, timeout=3600,
        )
        if result.returncode == 0:
            logger.info("[retrain] 재학습 완료")
            _notify_discord("✅ 주간 모델 재학습 완료 (v2)")
        else:
            logger.error("[retrain] 재학습 실패:\n%s", result.stderr[-2000:])
            _notify_discord(f"❌ 주간 모델 재학습 실패: {result.stderr[-200:]}")
    except subprocess.TimeoutExpired:
        logger.error("[retrain] 재학습 1시간 초과")
        _notify_discord("❌ 주간 모델 재학습 타임아웃")


# ──────────────────────────────────────────
# 2. 동적 워커 (DateTrigger 등록 대상)
# ──────────────────────────────────────────

def run_pre_game_sync(game_pk: int) -> None:
    """T-120min — Live Feed 1차 동기화 (라인업/날씨/선발)"""
    async def _inner() -> None:
        from src.collector.live_feed_client import LiveFeedClient
        async with LiveFeedClient() as lfc:
            with get_session() as session:
                await lfc.sync_to_db(game_pk, session)
        logger.info("[pre_game_sync] game %d 동기화 완료", game_pk)
    _run(f"pre_game_sync_{game_pk}", _inner)


def run_dynamic_inference(game_pk: int) -> None:
    """T-15min — Live Feed 재호출 + 47피처 추론 + LLM"""
    async def _inner() -> None:
        from scripts.run_inference_v3 import run_single
        await run_single(game_pk)
        _notify_discord(f"🎯 game {game_pk} 예측 저장 완료")
    _run(f"inference_{game_pk}", _inner)


def start_live_poller(game_pk: int) -> None:
    """T+0min — 1분 간격 라이브 폴러를 자가 종료형 IntervalTrigger 잡으로 등록"""
    sched = _get_global_scheduler()
    sched.add_job(
        _live_poll_tick,
        IntervalTrigger(minutes=1),
        args=[game_pk],
        id=f"live_{game_pk}",
        replace_existing=True,
        max_instances=1,
        misfire_grace_time=30,
    )
    logger.info("[live_poller] game %d 1분 polling 시작", game_pk)


def _live_poll_tick(game_pk: int) -> None:
    """1분마다 호출되는 라이브 폴러 워커. Final 감지 시 자가 제거 + postgame 트리거."""
    async def _inner() -> None:
        from src.collector.live_score_poller import LiveScorePoller
        async with LiveScorePoller() as p:
            with get_session() as session:
                final = await p.poll_once(game_pk, session)
        if final:
            sched = _get_global_scheduler()
            try:
                sched.remove_job(f"live_{game_pk}")
            except Exception:
                pass
            sched.add_job(
                run_postgame_sync,
                DateTrigger(run_date=datetime.now(ZoneInfo("Asia/Seoul")) + timedelta(minutes=5)),
                args=[game_pk],
                id=f"post_{game_pk}",
                replace_existing=True,
            )
            _notify_discord(f"🏁 game {game_pk} Final 감지 → 5분 후 postgame")
    _run(f"live_tick_{game_pk}", _inner)


def run_postgame_sync(game_pk: int) -> None:
    """경기 Final 직후 — boxscore → pitcher/batter game logs UPSERT"""
    async def _inner() -> None:
        from src.collector.postgame_collector import PostgameCollector
        async with PostgameCollector() as pc:
            with get_session() as session:
                await pc.sync_full(game_pk, session)
        _notify_discord(f"📦 game {game_pk} postgame 수집 완료")
    _run(f"postgame_{game_pk}", _inner)


# ──────────────────────────────────────────
# 3. 마스터 스케줄러 (13:00 KST)
# ──────────────────────────────────────────

def master_daily_scheduler(sched: BackgroundScheduler) -> None:
    """13:00 KST — 오늘 경기마다 T-120/T-15/T+0 워커를 DateTrigger 동적 등록"""
    async def _inner() -> None:
        today = date.today()
        url = (
            f"https://statsapi.mlb.com/api/v1/schedule"
            f"?sportId=1&date={today.isoformat()}"
        )
        async with aiohttp.ClientSession() as s:
            async with s.get(url, timeout=15) as r:
                data = await r.json()
        dates = data.get("dates", [])
        games = dates[0].get("games", []) if dates else []

        kst = ZoneInfo("Asia/Seoul")
        now = datetime.now(kst)
        registered = 0

        for g in games:
            pk = g["gamePk"]
            try:
                gd_utc = datetime.strptime(
                    g["gameDate"], "%Y-%m-%dT%H:%M:%SZ"
                ).replace(tzinfo=timezone.utc)
            except Exception:
                continue
            game_kst = gd_utc.astimezone(kst)
            sync_at = game_kst - timedelta(minutes=120)
            inf_at = game_kst - timedelta(minutes=15)
            live_at = game_kst

            if sync_at > now:
                sched.add_job(
                    run_pre_game_sync, DateTrigger(run_date=sync_at),
                    args=[pk], id=f"sync_{pk}", replace_existing=True,
                )
            if inf_at > now:
                sched.add_job(
                    run_dynamic_inference, DateTrigger(run_date=inf_at),
                    args=[pk], id=f"inf_{pk}", replace_existing=True,
                )
            if live_at > now:
                sched.add_job(
                    start_live_poller, DateTrigger(run_date=live_at),
                    args=[pk], id=f"livestart_{pk}", replace_existing=True,
                )
                registered += 1

        _notify_discord(
            f"👑 마스터: 오늘 {registered}경기 (pre/inf/live) 동적 예약 완료"
        )
    _run("master_daily_scheduler", _inner)


# ──────────────────────────────────────────
# 헬퍼
# ──────────────────────────────────────────

def _run(name: str, coro_factory) -> None:
    try:
        logger.info("[pipeline] Starting %s", name)
        asyncio.run(coro_factory())
    except Exception as e:
        logger.error("[pipeline] %s FAILED: %s", name, e, exc_info=True)
        _notify_discord(f"❌ {name} 실패: {e}")


def _notify_discord(message: str) -> None:
    if not settings.discord_webhook_url:
        return
    try:
        import json
        import urllib.request
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
# 4. 스케줄러 부트스트랩
# ──────────────────────────────────────────

_SCHED: BackgroundScheduler | None = None


def _get_global_scheduler() -> BackgroundScheduler:
    """라이브 폴러가 자기 자신 제거하기 위해 module-level scheduler 참조."""
    assert _SCHED is not None, "setup_scheduler() must be called first"
    return _SCHED


def setup_scheduler() -> BackgroundScheduler:
    global _SCHED
    sched = BackgroundScheduler(timezone="Asia/Seoul")

    sched.add_job(run_bref_daily_update,   CronTrigger(hour=6,  minute=30))
    sched.add_job(run_morning_pipeline,    CronTrigger(hour=7,  minute=0))
    sched.add_job(run_player_stats_update, CronTrigger(hour=7,  minute=30))
    sched.add_job(run_statcast_pipeline,   CronTrigger(hour=12, minute=0))
    sched.add_job(run_fangraphs_update,    CronTrigger(hour=12, minute=30))
    sched.add_job(
        master_daily_scheduler, CronTrigger(hour=13, minute=0), args=[sched]
    )
    sched.add_job(run_inference_fallback,  CronTrigger(hour=19, minute=30))
    sched.add_job(
        retrain_models_task,
        CronTrigger(day_of_week="sun", hour=3, minute=0),
    )

    _SCHED = sched
    return sched


if __name__ == "__main__":
    scheduler = setup_scheduler()
    scheduler.start()

    # 부팅 시 오늘 일정 1회 강제 등록
    master_daily_scheduler(scheduler)

    logger.info("Scheduler started. Press Ctrl+C to exit.")
    try:
        import time
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        scheduler.shutdown()

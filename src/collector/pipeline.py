"""APScheduler 기반 일일 자동 파이프라인 — 현지 변수 방어형 (타임존: America/New_York 기준)

타임라인 변환 (서머타임 기준):
  - ET 03:00 (KST 16:00) [일] 주간 모델 재학습
  - ET 04:30 (KST 17:30) BBref 스텔스 스크래퍼 (전일 완벽 마감 후 덮어쓰기)
  - ET 05:00 (KST 18:00) run_morning_pipeline & 마스터 스케줄러 동시 실행
  - ET 05:30 (KST 18:30) 전일 Statcast (1일 delta)
  - ET 11:30 (KST 00:30) fallback (동적 워커 누락 시 일괄 보충)
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
from src.collector.mlb_statsapi_client import MLBStatsAPIClient
from src.collector.statcast_collector import fetch_statcast_range, save_statcast
from src.common.logger import get_logger
from src.db.session import get_session

logger = get_logger(__name__)

# 글로벌 타임존 정의 (미국 동부 시간 기준)
TARGET_TZ = ZoneInfo("America/New_York")


# ──────────────────────────────────────────
# 1. 고정 스케줄 태스크 (타임라인 최적화)
# ──────────────────────────────────────────

def run_morning_pipeline() -> None:
    """ET 05:00 (KST 18:00) — 전일 결과 완벽 마감 후 동기화"""
    async def _inner() -> None:
        # 미국 동부 시간 기준 어제와 오늘 정의
        today_et = datetime.now(TARGET_TZ).date()
        yesterday_et = today_et - timedelta(days=1)
        
        async with MLBStatsAPIClient() as client:
            with get_session() as session:
                await client.update_game_results(yesterday_et, session)
                await client.sync_schedule(today_et, session)
        _notify_discord(
            f"✅ [Morning Pipeline] {yesterday_et} 결과 + {today_et} 일정 동기화 완료"
        )
    _run(run_morning_pipeline.__name__, _inner)


def run_bref_daily_update() -> None:
    """ET 04:30 (KST 17:30) — BBref 전일 최종 데이터 마감 후 갱신"""
    def _inner() -> None:
        from src.collector.bref_scraper import update_bref_season
        today_et = datetime.now(TARGET_TZ).date()
        counts = update_bref_season(today_et.year)
        _notify_discord(
            f"✅ [BBref] pitching:{counts.get('pitching', 0)} batting:{counts.get('batting', 0)}"
        )
    try:
        logger.info("[pipeline] Starting run_bref_daily_update")
        _inner()
    except Exception as e:
        logger.error("[pipeline] run_bref_daily_update FAILED: %s", e, exc_info=True)
        _notify_discord(f"❌ BBref 업데이트 실패: {e}")


def run_statcast_pipeline() -> None:
    """ET 05:30 (KST 18:30) — 서부 연장전까지 완벽 반영된 전일 Statcast 수집"""
    async def _inner() -> None:
        yesterday_et = datetime.now(TARGET_TZ).date() - timedelta(days=1)
        with get_session() as session:
            df = await fetch_statcast_range(yesterday_et, yesterday_et)
            count = await save_statcast(df, session)
        _notify_discord(f"✅ [Statcast] {yesterday_et} 데이터 {count}개 저장 완료")
    _run(run_statcast_pipeline.__name__, _inner)


def run_inference_fallback() -> None:
    """ET 11:30 (KST 00:30) 폴백 — 현지 시간 오전 중 누락 경기 보충"""
    async def _inner() -> None:
        from scripts.run_inference_v3 import run_all_today
        await run_all_today()
        _notify_discord("✅ [Fallback] 당일 추론 보충 완료")
    _run("inference_fallback", _inner)


def retrain_models_task() -> None:
    """ET 일요일 03:00 (KST 16:00) — 주간 모델 재학습"""
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
            _notify_discord(f"❌ 주간 모델 재학습 실패: result.stderr[-200:]")
    except subprocess.TimeoutExpired:
        logger.error("[retrain] 재학습 1시간 초과")
        _notify_discord("❌ 주간 모델 재학습 타임아웃")


# ──────────────────────────────────────────
# 2. 동적 워커 (DateTrigger 등록 대상 — 수정 없음)
# ──────────────────────────────────────────

def run_pre_game_sync(game_pk: int) -> None:
    async def _inner() -> None:
        from src.collector.live_feed_client import LiveFeedClient
        async with LiveFeedClient() as lfc:
            with get_session() as session:
                await lfc.sync_to_db(game_pk, session)
        logger.info("[pre_game_sync] game %d 동기화 완료", game_pk)
    _run(f"pre_game_sync_{game_pk}", _inner)


def run_dynamic_inference(game_pk: int) -> None:
    async def _inner() -> None:
        from scripts.run_inference_v3 import run_single
        await run_single(game_pk)
        _notify_discord(f"🎯 game {game_pk} 예측 저장 완료")
    _run(f"inference_{game_pk}", _inner)


def start_live_poller(game_pk: int) -> None:
    sched = _get_global_scheduler()
    sched.add_job(
        _live_poll_tick,
        IntervalTrigger(seconds=30),
        args=[game_pk],
        id=f"live_{game_pk}",
        replace_existing=True,
        max_instances=1,
        misfire_grace_time=15,
    )
    logger.info("[live_poller] game %d 30초 polling 시작", game_pk)


def _live_poll_tick(game_pk: int) -> None:
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
            # Postgame 수집도 스케줄러 타임존에 맞춰 실행되도록 유도
            run_time = datetime.now(TARGET_TZ) + timedelta(minutes=5)
            sched.add_job(
                run_postgame_sync,
                DateTrigger(run_date=run_time),
                args=[game_pk],
                id=f"post_{game_pk}",
                replace_existing=True,
            )
            _notify_discord(f"🏁 game {game_pk} Final 감지 → 5분 후 postgame")
    _run(f"live_tick_{game_pk}", _inner)


def run_postgame_sync(game_pk: int) -> None:
    async def _inner() -> None:
        from src.collector.postgame_collector import PostgameCollector
        async with PostgameCollector() as pc:
            with get_session() as session:
                await pc.sync_full(game_pk, session)
        _notify_discord(f"📦 game {game_pk} postgame 수집 완료")
    _run(f"postgame_{game_pk}", _inner)


# ──────────────────────────────────────────
# 3. 마스터 스케줄러 (ET 05:00 / KST 18:00)
# ──────────────────────────────────────────

def master_daily_scheduler(sched: BackgroundScheduler) -> None:
    """ET 05:00 (KST 18:00) — 오늘(현지 날짜) 열릴 모든 경기를 스캔하여 동적 예약"""
    async def _inner() -> None:
        today_et = datetime.now(TARGET_TZ).date()
        url = (
            f"https://statsapi.mlb.com/api/v1/schedule"
            f"?sportId=1&date={today_et.isoformat()}"
        )
        async with aiohttp.ClientSession() as s:
            async with s.get(url, timeout=15) as r:
                data = await r.json()
        dates = data.get("dates", [])
        games = dates[0].get("games", []) if dates else []

        now_et = datetime.now(TARGET_TZ)
        registered = 0

        for g in games:
            pk = g["gamePk"]
            try:
                # API의 UTC 시간을 타임존 객체로 변환한 뒤 America/New_York 타임존으로 통합
                gd_utc = datetime.strptime(
                    g["gameDate"], "%Y-%m-%dT%H:%M:%SZ"
                ).replace(tzinfo=timezone.utc)
            except Exception:
                continue
            
            game_et = gd_utc.astimezone(TARGET_TZ)
            sync_at = game_et - timedelta(minutes=120)
            inf_at = game_et - timedelta(minutes=15)
            live_at = game_et

            # 스케줄러 타임존(TARGET_TZ) 스탬프를 기반으로 Job 추가
            if sync_at > now_et:
                sched.add_job(
                    run_pre_game_sync, DateTrigger(run_date=sync_at),
                    args=[pk], id=f"sync_{pk}", replace_existing=True,
                )
            if inf_at > now_et:
                sched.add_job(
                    run_dynamic_inference, DateTrigger(run_date=inf_at),
                    args=[pk], id=f"inf_{pk}", replace_existing=True,
                )
            if live_at > now_et:
                sched.add_job(
                    start_live_poller, DateTrigger(run_date=live_at),
                    args=[pk], id=f"livestart_{pk}", replace_existing=True,
                )
                registered += 1

        _notify_discord(
            f"👑 [Master] {today_et} 일정 스캔 완료: 총 {registered}경기 동적 워커 등록 완료"
        )
    _run("master_daily_scheduler", _inner)


# ──────────────────────────────────────────
# 4. 시작 시 미처리 경기 즉시 처리
# ──────────────────────────────────────────

def run_startup_catchup(sched: BackgroundScheduler) -> None:
    """스케줄러 시작 시 이미 종료/진행 중인 당일 경기를 즉시 처리.
    - Final 경기: update_game_results로 스코어 + is_correct 일괄 갱신 후 Redis 캐시 삭제
    - In Progress 경기: 라이브 폴러 즉시 등록
    """
    async def _inner() -> None:
        now_et = datetime.now(TARGET_TZ)
        today_et = now_et.date()
        # ET 자정 이후(~09:00 ET)는 어제 경기가 Final 상태 — yesterday_et로 조회
        # 낮/저녁 시간대(09:00 ET~)는 today_et로 조회
        # 두 날짜 모두 스캔해서 Final/Live 경기를 모두 잡는다
        check_dates = list({today_et, today_et - timedelta(days=1)})

        all_finished: list = []
        all_live: list = []
        finished_date: date | None = None

        for check_date in check_dates:
            url = (
                f"https://statsapi.mlb.com/api/v1/schedule"
                f"?sportId=1&date={check_date.isoformat()}"
            )
            async with aiohttp.ClientSession() as s:
                async with s.get(url, timeout=15) as r:
                    data = await r.json()
            dates = data.get("dates", [])
            games = dates[0].get("games", []) if dates else []

            fin = [g for g in games if g.get("status", {}).get("abstractGameState") == "Final"]
            liv = [g for g in games if g.get("status", {}).get("abstractGameState") == "Live"]

            if fin and not all_finished:
                all_finished = fin
                finished_date = check_date
            all_live.extend(liv)

        logger.info(
            "[startup_catchup] 스캔 완료 — Final %d경기 (ET %s), Live %d경기",
            len(all_finished), finished_date, len(all_live),
        )

        # 1. 종료 경기 일괄 갱신
        if all_finished and finished_date is not None:
            async with MLBStatsAPIClient() as client:
                with get_session() as session:
                    await client.update_game_results(finished_date, session)
            logger.info(
                "[startup_catchup] %d 종료 경기 결과 갱신 완료", len(all_finished)
            )
            # Redis 캐시 삭제 (KST 날짜 = ET날짜 + 1일)
            kst_date = (finished_date + timedelta(days=1)).isoformat()
            try:
                import redis as sync_redis
                rc = sync_redis.from_url(
                    f"redis://{settings.redis_host}:{settings.redis_port}"
                )
                rc.delete(f"predictions:today:{kst_date}")
                rc.delete(f"archive:{kst_date}")
                rc.close()
                logger.info("[startup_catchup] Redis 캐시 삭제: %s", kst_date)
            except Exception as e:
                logger.warning("[startup_catchup] Redis 캐시 삭제 실패: %s", e)

        # 2. 진행 중 경기 라이브 폴러 즉시 등록
        for g in all_live:
            pk = g["gamePk"]
            if not sched.get_job(f"live_{pk}"):
                start_live_poller(pk)
                logger.info("[startup_catchup] game %d 라이브 폴러 즉시 시작", pk)

        _notify_discord(
            f"🔄 [Startup Catchup] 종료 {len(all_finished)}경기 갱신"
            f", 진행중 {len(all_live)}경기 폴러 등록"
        )
    _run("startup_catchup", _inner)


# ──────────────────────────────────────────
# 헬퍼 함수 및 부트스트랩
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


_SCHED: BackgroundScheduler | None = None


def _get_global_scheduler() -> BackgroundScheduler:
    assert _SCHED is not None, "setup_scheduler() must be called first"
    return _SCHED


def setup_scheduler() -> BackgroundScheduler:
    global _SCHED
    
    # 🌟 CRITICAL: 스케줄러 타임존을 미 동부 시간으로 변경하여 서머타임 이슈 원천 차단
    sched = BackgroundScheduler(timezone=TARGET_TZ)

    # 미 동부 시간 기준 크론 트리거 등록 (현지 새벽 시간대 일괄 배치)
    sched.add_job(run_bref_daily_update,   CronTrigger(hour=4,  minute=30))
    sched.add_job(run_morning_pipeline,    CronTrigger(hour=5,  minute=0))
    sched.add_job(run_statcast_pipeline,   CronTrigger(hour=5,  minute=30))
    sched.add_job(
        master_daily_scheduler, CronTrigger(hour=5, minute=0), args=[sched]
    )
    sched.add_job(run_inference_fallback,  CronTrigger(hour=11, minute=30))
    sched.add_job(
        retrain_models_task,
        CronTrigger(day_of_week="sun", hour=3, minute=0),
    )

    _SCHED = sched
    return sched


if __name__ == "__main__":
    scheduler = setup_scheduler()
    scheduler.start()

    # 미래 경기 동적 워커 등록
    master_daily_scheduler(scheduler)
    # 이미 종료/진행 중인 경기 즉시 처리
    run_startup_catchup(scheduler)

    logger.info("Scheduler started with America/New_York timezone. Press Ctrl+C to exit.")
    try:
        import time
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        scheduler.shutdown()
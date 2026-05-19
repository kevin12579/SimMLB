"""2023~2024 시즌 과거 데이터 백필 스크립트 (Week 1 금요일 야간 실행)"""
import asyncio
import sys
from datetime import date, timedelta

sys.path.insert(0, ".")

from src.collector.mlb_statsapi_client import MLBStatsAPIClient
from src.collector.statcast_collector import fetch_statcast_range, save_statcast
from src.collector.fangraphs_collector import (
    fetch_pitching_stats, fetch_batting_stats,
    save_pitching_stats, save_batting_stats,
)
from src.collector.roster_sync import sync_all_rosters
from src.db.session import get_session
from src.common.logger import get_logger

logger = get_logger("backfill")

SEASON_DATES = {
    2023: (date(2023, 3, 30), date(2023, 10, 1)),
    2024: (date(2024, 3, 20), date(2024, 9, 29)),
}


async def backfill_teams_and_rosters(session_factory: object) -> None:
    async with MLBStatsAPIClient() as client:
        with get_session() as session:
            await client.sync_teams(session)
            logger.info("Teams synced")

    for season in SEASON_DATES:
        with get_session() as session:
            await sync_all_rosters(season, session)
            logger.info("Rosters synced for %d", season)


async def backfill_schedules(season: int) -> None:
    start, end = SEASON_DATES[season]
    current = start
    async with MLBStatsAPIClient() as client:
        while current <= end:
            with get_session() as session:
                try:
                    await client.sync_schedule(current, session)
                except Exception as e:
                    logger.warning("Schedule sync failed for %s: %s", current, e)
            current += timedelta(days=1)
            await asyncio.sleep(0.5)
    logger.info("Schedule backfill done for %d", season)


async def backfill_game_results(season: int) -> None:
    start, end = SEASON_DATES[season]
    current = start
    async with MLBStatsAPIClient() as client:
        while current <= end:
            with get_session() as session:
                try:
                    await client.update_game_results(current, session)
                except Exception as e:
                    logger.warning("Result update failed for %s: %s", current, e)
            current += timedelta(days=1)
            await asyncio.sleep(0.3)
    logger.info("Results backfill done for %d", season)


async def backfill_statcast(season: int) -> None:
    start, end = SEASON_DATES[season]
    # 월 단위로 분할
    current = start.replace(day=1)
    while current <= end:
        month_end = (current.replace(day=28) + timedelta(days=4)).replace(day=1) - timedelta(days=1)
        chunk_end = min(month_end, end)
        with get_session() as session:
            try:
                df = await fetch_statcast_range(current, chunk_end)
                count = await save_statcast(df, session)
                logger.info("Statcast %s~%s: %d pitches saved", current, chunk_end, count)
            except Exception as e:
                logger.error("Statcast backfill failed %s~%s: %s", current, chunk_end, e)
        current = chunk_end + timedelta(days=1)
        await asyncio.sleep(5)


async def backfill_fangraphs(season: int) -> None:
    as_of = SEASON_DATES[season][1]
    with get_session() as session:
        pitch_df = await fetch_pitching_stats(season)
        await save_pitching_stats(pitch_df, season, as_of, session)
        bat_df = await fetch_batting_stats(season)
        await save_batting_stats(bat_df, season, as_of, session)
    logger.info("FanGraphs backfill done for %d", season)


async def backfill(start_season: int = 2023, end_season: int = 2024) -> None:
    logger.info("=== 백필 시작: %d ~ %d ===", start_season, end_season)

    # 1. 팀 + 로스터
    await backfill_teams_and_rosters(None)

    for season in range(start_season, end_season + 1):
        logger.info("--- Season %d ---", season)

        # 2. 경기 일정
        await backfill_schedules(season)

        # 3. 경기 결과
        await backfill_game_results(season)

        # 4. FanGraphs
        await backfill_fangraphs(season)

        # 5. Statcast (시간이 가장 오래 걸림)
        await backfill_statcast(season)

    logger.info("=== 백필 완료 ===")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="MLB 과거 데이터 백필")
    parser.add_argument("--start", type=int, default=2023)
    parser.add_argument("--end",   type=int, default=2024)
    args = parser.parse_args()
    asyncio.run(backfill(args.start, args.end))

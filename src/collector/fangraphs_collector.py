import asyncio
from datetime import date

import pandas as pd
from sqlalchemy.orm import Session
from sqlalchemy.dialects.postgresql import insert

from src.db.models.players import PlayerSeasonStats
from src.common.logger import get_logger

logger = get_logger(__name__)


def _fetch_pitching_leaders_sync(season: int) -> pd.DataFrame:
    import pybaseball
    return pybaseball.pitching_stats(season, season, qual=1)


def _fetch_batting_leaders_sync(season: int) -> pd.DataFrame:
    import pybaseball
    return pybaseball.batting_stats(season, season, qual=1)


async def fetch_pitching_stats(season: int) -> pd.DataFrame:
    logger.info("Fetching FanGraphs pitching stats %d", season)
    return await asyncio.to_thread(_fetch_pitching_leaders_sync, season)


async def fetch_batting_stats(season: int) -> pd.DataFrame:
    logger.info("Fetching FanGraphs batting stats %d", season)
    return await asyncio.to_thread(_fetch_batting_leaders_sync, season)


def _safe(val: object) -> float | None:
    try:
        f = float(val)  # type: ignore[arg-type]
        return f if f == f else None  # NaN check
    except (TypeError, ValueError):
        return None


async def save_pitching_stats(df: pd.DataFrame, season: int, as_of: date, session: Session) -> int:
    """FanGraphs 투수 통계를 player_season_stats에 저장"""
    if df is None or df.empty:
        return 0

    # FanGraphs mlbam id 열: 'IDfg' or 'MLBAMID' 등 버전마다 다름
    id_col = next((c for c in ["MLBAMID", "mlbam_id", "xMLBAMID"] if c in df.columns), None)
    if id_col is None:
        logger.warning("No MLBAM ID column found in pitching stats")
        return 0

    rows = []
    for _, row in df.iterrows():
        mlbam_id = row.get(id_col)
        if not mlbam_id or pd.isna(mlbam_id):
            continue
        rows.append(dict(
            player_id=int(mlbam_id),
            season=season,
            as_of_date=as_of,
            era=_safe(row.get("ERA")),
            fip=_safe(row.get("FIP")),
            xfip=_safe(row.get("xFIP")),
            whip=_safe(row.get("WHIP")),
            k_pct=_safe(row.get("K%")),
            bb_pct=_safe(row.get("BB%")),
            hr9=_safe(row.get("HR/9")),
            innings_pitched=_safe(row.get("IP")),
        ))

    if not rows:
        return 0
    session.execute(insert(PlayerSeasonStats).values(rows).on_conflict_do_nothing())
    session.commit()
    logger.info("Saved %d pitcher season stats", len(rows))
    return len(rows)


async def save_batting_stats(df: pd.DataFrame, season: int, as_of: date, session: Session) -> int:
    """FanGraphs 타자 통계를 player_season_stats에 저장"""
    if df is None or df.empty:
        return 0

    id_col = next((c for c in ["MLBAMID", "mlbam_id", "xMLBAMID"] if c in df.columns), None)
    if id_col is None:
        logger.warning("No MLBAM ID column found in batting stats")
        return 0

    rows = []
    for _, row in df.iterrows():
        mlbam_id = row.get(id_col)
        if not mlbam_id or pd.isna(mlbam_id):
            continue
        rows.append(dict(
            player_id=int(mlbam_id),
            season=season,
            as_of_date=as_of,
            wrc_plus=_safe(row.get("wRC+")),
            ops=_safe(row.get("OPS")),
            babip=_safe(row.get("BABIP")),
        ))

    if not rows:
        return 0
    session.execute(insert(PlayerSeasonStats).values(rows).on_conflict_do_nothing())
    session.commit()
    logger.info("Saved %d batter season stats", len(rows))
    return len(rows)

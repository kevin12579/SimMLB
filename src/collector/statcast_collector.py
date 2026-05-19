import asyncio
from datetime import date, timedelta

import pandas as pd
from sqlalchemy.orm import Session
from sqlalchemy.dialects.postgresql import insert

from src.db.models.games import StatcastPitch
from src.common.logger import get_logger
from src.common.exceptions import DataCollectionError
from src.db.player_utils import ensure_players_exist

logger = get_logger(__name__)

CHUNK_DAYS = 5  # baseballsavant 502 방지용 청크 크기


def _fetch_statcast_sync(start_dt: str, end_dt: str) -> pd.DataFrame:
    """pybaseball 동기 함수 — asyncio.to_thread로 래핑해서 사용"""
    import pybaseball
    pybaseball.cache.enable()
    return pybaseball.statcast(start_dt=start_dt, end_dt=end_dt, verbose=False)


async def fetch_statcast_range(start: date, end: date) -> pd.DataFrame:
    """날짜 범위 Statcast 수집 (5일 단위 청크)"""
    all_frames = []
    current = start
    while current <= end:
        chunk_end = min(current + timedelta(days=CHUNK_DAYS - 1), end)
        logger.info("Fetching Statcast %s ~ %s", current, chunk_end)
        try:
            df = await asyncio.to_thread(
                _fetch_statcast_sync,
                current.strftime("%Y-%m-%d"),
                chunk_end.strftime("%Y-%m-%d"),
            )
            if df is not None and not df.empty:
                all_frames.append(df)
        except Exception as e:
            logger.warning("Statcast chunk %s~%s failed: %s", current, chunk_end, e)
        current = chunk_end + timedelta(days=1)
        await asyncio.sleep(2)  # baseballsavant rate limit 방지

    if not all_frames:
        return pd.DataFrame()
    return pd.concat(all_frames, ignore_index=True)


def _enrich(df: pd.DataFrame) -> pd.DataFrame:
    """is_hard_hit, is_barrel, swing, whiff 컬럼 추가"""
    df = df.copy()
    df["is_hard_hit"] = (df.get("launch_speed", pd.Series(dtype=float)) >= 95).fillna(False)
    # barrel: launch_angle 26~30도 + launch_speed >= 98mph
    la = df.get("launch_angle", pd.Series(dtype=float))
    ls = df.get("launch_speed", pd.Series(dtype=float))
    df["is_barrel"] = ((la >= 26) & (la <= 30) & (ls >= 98)).fillna(False)
    swing_events = {"swinging_strike", "swinging_strike_blocked", "foul", "foul_tip",
                    "hit_into_play", "hit_into_play_no_out", "hit_into_play_score"}
    df["swing"] = df.get("description", pd.Series(dtype=str)).isin(swing_events)
    df["whiff"] = df.get("description", pd.Series(dtype=str)).isin(
        {"swinging_strike", "swinging_strike_blocked"}
    )
    return df


async def save_statcast(df: pd.DataFrame, session: Session) -> int:
    """Statcast 데이터를 DB에 저장, 저장된 행 수 반환"""
    if df.empty:
        return 0

    df = _enrich(df)

    # pitcher/batter FK 위반 방지: 없는 선수 ID를 stub으로 사전 삽입
    player_ids: set[int] = set()
    for col in ("pitcher", "batter"):
        if col in df.columns:
            player_ids.update(int(v) for v in df[col].dropna().unique())
    ensure_players_exist(player_ids, session)

    rows = []
    for _, row in df.iterrows():
        rows.append(dict(
            game_date=row.get("game_date"),
            game_pk=row.get("game_pk"),
            pitcher_id=row.get("pitcher"),
            batter_id=row.get("batter"),
            pitch_type=str(row.get("pitch_type", ""))[:5] or None,
            release_speed=row.get("release_speed") if pd.notna(row.get("release_speed")) else None,
            spin_rate=row.get("release_spin_rate") if pd.notna(row.get("release_spin_rate")) else None,
            launch_speed=row.get("launch_speed") if pd.notna(row.get("launch_speed")) else None,
            launch_angle=row.get("launch_angle") if pd.notna(row.get("launch_angle")) else None,
            is_hard_hit=bool(row.get("is_hard_hit", False)),
            is_barrel=bool(row.get("is_barrel", False)),
            events=str(row.get("events", ""))[:50] or None,
            description=str(row.get("description", ""))[:50] or None,
            swing=bool(row.get("swing", False)),
            whiff=bool(row.get("whiff", False)),
        ))

    # 배치 insert (충돌 무시)
    BATCH = 500
    saved = 0
    for i in range(0, len(rows), BATCH):
        batch = rows[i:i + BATCH]
        session.execute(insert(StatcastPitch).values(batch).on_conflict_do_nothing())
        saved += len(batch)
    session.commit()
    logger.info("Saved %d statcast pitches", saved)
    return saved

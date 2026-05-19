"""타자/팀 Statcast 피처 (8개) + 라인업 wRC+ (3개)"""
from datetime import date, timedelta

from sqlalchemy import text
from sqlalchemy.orm import Session


def get_batter_features(team_id: int, as_of_date: date, is_home: bool, session: Session) -> dict:
    prefix = "home" if is_home else "away"
    window_start = as_of_date - timedelta(days=14)

    # 팀 Statcast 집계 (최근 14일)
    sc_row = session.execute(text("""
        SELECT
            AVG(CASE WHEN is_barrel THEN 1.0 ELSE 0.0 END) AS barrel_rate,
            AVG(xwoba) AS xwoba,
            AVG(CASE WHEN is_hard_hit THEN 1.0 ELSE 0.0 END) AS hard_hit_pct,
            SUM(CASE WHEN NOT swing AND description NOT LIKE '%ball%' THEN 1 ELSE 0 END)::float
              / NULLIF(COUNT(*), 0) AS chase_rate
        FROM statcast_pitches
        WHERE batter_team_id = :tid
          AND game_date >= :start
          AND game_date < :as_of
    """), {"tid": team_id, "start": window_start, "as_of": as_of_date}).fetchone()

    barrel_rate   = float(sc_row.barrel_rate  or 0.06)
    xwoba         = float(sc_row.xwoba        or 0.320)
    hard_hit_pct  = float(sc_row.hard_hit_pct or 0.36)
    chase_rate    = float(sc_row.chase_rate   or 0.30)

    # 라인업 wRC+ (game_lineups에서 확정 라인업 9명 조회)
    lineup_row = session.execute(text("""
        SELECT AVG(pss.wrc_plus) AS lineup_wrc_plus
        FROM game_lineups gl
        JOIN player_season_stats pss ON pss.player_id = gl.player_id
        WHERE gl.team_id = :tid
          AND gl.game_date = :as_of
          AND gl.is_home = :is_home
          AND pss.as_of_date = (
              SELECT MAX(as_of_date) FROM player_season_stats
              WHERE player_id = gl.player_id
                AND as_of_date < :as_of
          )
    """), {"tid": team_id, "as_of": as_of_date, "is_home": is_home}).fetchone()

    # game_lineups에 is_home 컬럼이 date 형이 아니므로 간단하게 대체
    lineup_wrc_plus = 100.0
    if lineup_row and lineup_row.lineup_wrc_plus:
        lineup_wrc_plus = float(lineup_row.lineup_wrc_plus)

    return {
        f"{prefix}_team_barrel_rate":  barrel_rate,
        f"{prefix}_team_xwoba":        xwoba,
        f"{prefix}_team_hard_hit_pct": hard_hit_pct,
        f"{prefix}_team_chase_rate":   chase_rate,
        f"{prefix}_lineup_wrc_plus":   lineup_wrc_plus,
    }

"""선발 투수 피처 (12개) — as_of_date 이전 데이터만 사용"""
from datetime import date

from sqlalchemy import text
from sqlalchemy.orm import Session

DEFAULT_ERA = 4.50


def get_pitcher_features(pitcher_id: int | None, as_of_date: date, is_home: bool, session: Session) -> dict:
    prefix = "home_sp" if is_home else "away_sp"

    if pitcher_id is None:
        return _default_pitcher_features(prefix)

    # 시즌 ERA (pitcher_game_logs 누적)
    era_row = session.execute(text("""
        SELECT SUM(er)*9.0/NULLIF(SUM(ip), 0) AS era,
               SUM(k)::float/NULLIF(SUM(ip)*3, 0) AS k_per_ip,
               SUM(bb)::float/NULLIF(SUM(ip)*3, 0) AS bb_per_ip
        FROM pitcher_game_logs
        WHERE player_id = :pid
          AND game_date < :as_of
          AND is_starter = true
          AND EXTRACT(YEAR FROM game_date) = EXTRACT(YEAR FROM :as_of::date)
    """), {"pid": pitcher_id, "as_of": as_of_date}).fetchone()

    season_era = float(era_row.era) if era_row and era_row.era else DEFAULT_ERA
    k_pct      = float(era_row.k_per_ip) if era_row and era_row.k_per_ip else 0.20
    bb_pct     = float(era_row.bb_per_ip) if era_row and era_row.bb_per_ip else 0.08

    # Last 3 선발 ERA
    last3_row = session.execute(text("""
        SELECT SUM(er)*9.0/NULLIF(SUM(ip), 0) AS last3_era
        FROM (
            SELECT er, ip FROM pitcher_game_logs
            WHERE player_id = :pid
              AND game_date < :as_of
              AND is_starter = true
            ORDER BY game_date DESC LIMIT 3
        ) recent
    """), {"pid": pitcher_id, "as_of": as_of_date}).fetchone()
    last3_era = float(last3_row.last3_era) if last3_row and last3_row.last3_era else season_era

    # FanGraphs FIP / xFIP
    fg_row = session.execute(text("""
        SELECT fip, xfip, whip
        FROM player_season_stats
        WHERE player_id = :pid
          AND as_of_date < :as_of
          AND EXTRACT(YEAR FROM as_of_date) = EXTRACT(YEAR FROM :as_of::date)
        ORDER BY as_of_date DESC LIMIT 1
    """), {"pid": pitcher_id, "as_of": as_of_date}).fetchone()
    fip  = float(fg_row.fip)  if fg_row and fg_row.fip  else season_era
    xfip = float(fg_row.xfip) if fg_row and fg_row.xfip else season_era
    whip = float(fg_row.whip) if fg_row and fg_row.whip else 1.30

    # Statcast: 헛스윙률
    sc_row = session.execute(text("""
        SELECT
            SUM(CASE WHEN whiff THEN 1 ELSE 0 END)::float / NULLIF(SUM(CASE WHEN swing THEN 1 ELSE 0 END), 0) AS swstr
        FROM statcast_pitches
        WHERE pitcher_id = :pid
          AND game_date >= :as_of::date - INTERVAL '30 days'
          AND game_date < :as_of
    """), {"pid": pitcher_id, "as_of": as_of_date}).fetchone()
    swstr_pct = float(sc_row.swstr) if sc_row and sc_row.swstr else 0.10

    return {
        f"{prefix}_season_era":  season_era,
        f"{prefix}_fip":         fip,
        f"{prefix}_xfip":        xfip,
        f"{prefix}_last3_era":   last3_era,
        f"{prefix}_k_pct":       k_pct,
        f"{prefix}_bb_pct":      bb_pct,
        f"{prefix}_whip":        whip,
        f"{prefix}_swstr_pct":   swstr_pct,
    }


def _default_pitcher_features(prefix: str) -> dict:
    return {
        f"{prefix}_season_era":  DEFAULT_ERA,
        f"{prefix}_fip":         DEFAULT_ERA,
        f"{prefix}_xfip":        DEFAULT_ERA,
        f"{prefix}_last3_era":   DEFAULT_ERA,
        f"{prefix}_k_pct":       0.20,
        f"{prefix}_bb_pct":      0.08,
        f"{prefix}_whip":        1.30,
        f"{prefix}_swstr_pct":   0.10,
    }

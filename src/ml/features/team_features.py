"""팀 누적 지표 피처 (13개) — as_of_date 이전 데이터만 사용"""
from datetime import date

from sqlalchemy import text
from sqlalchemy.orm import Session


def shrunk_rate(wins: int, games: int, beta: float = 20, prior: float = 0.5) -> float:
    """소표본일수록 리그 평균(0.5)으로 수렴"""
    if games == 0:
        return prior
    return (wins + beta * prior) / (games + beta)


def pythagenpat_wp(rs: int, ra: int, games: int) -> float:
    if games == 0 or (rs + ra) == 0:
        return 0.5
    rpg = (rs + ra) / games
    x = rpg ** 0.287
    return (rs ** x) / (rs ** x + ra ** x)


def get_team_features(team_id: int, as_of_date: date, is_home: bool, session: Session) -> dict:
    """팀 피처 13개 계산"""
    prefix = "home" if is_home else "away"

    # 시즌 전체 승률
    season_row = session.execute(text("""
        SELECT
            SUM(CASE WHEN home_team_id = :tid AND home_score > away_score THEN 1
                     WHEN away_team_id = :tid AND away_score > home_score THEN 1
                     ELSE 0 END) AS wins,
            COUNT(*) AS games,
            SUM(CASE WHEN home_team_id = :tid THEN home_score
                     WHEN away_team_id = :tid THEN away_score END) AS rs,
            SUM(CASE WHEN home_team_id = :tid THEN away_score
                     WHEN away_team_id = :tid THEN home_score END) AS ra
        FROM games
        WHERE (home_team_id = :tid OR away_team_id = :tid)
          AND game_date < :as_of
          AND status = 'Final'
          AND EXTRACT(YEAR FROM game_date) = EXTRACT(YEAR FROM :as_of::date)
    """), {"tid": team_id, "as_of": as_of_date}).fetchone()

    wins  = int(season_row.wins  or 0)
    games = int(season_row.games or 0)
    rs    = int(season_row.rs    or 0)
    ra    = int(season_row.ra    or 0)

    season_win_rate   = shrunk_rate(wins, games)
    pyth_wp           = pythagenpat_wp(rs, ra, games)

    # 홈경기 승률
    home_row = session.execute(text("""
        SELECT
            SUM(CASE WHEN home_score > away_score THEN 1 ELSE 0 END) AS hw,
            COUNT(*) AS hg
        FROM games
        WHERE home_team_id = :tid
          AND game_date < :as_of
          AND status = 'Final'
          AND EXTRACT(YEAR FROM game_date) = EXTRACT(YEAR FROM :as_of::date)
    """), {"tid": team_id, "as_of": as_of_date}).fetchone()
    hw = int(home_row.hw or 0)
    hg = int(home_row.hg or 0)
    home_win_rate = shrunk_rate(hw, hg)

    # 최근 10경기
    last10 = session.execute(text("""
        SELECT
            SUM(CASE WHEN home_team_id = :tid AND home_score > away_score THEN 1
                     WHEN away_team_id = :tid AND away_score > home_score THEN 1
                     ELSE 0 END) AS w10,
            SUM(CASE WHEN home_team_id = :tid THEN home_score
                     WHEN away_team_id = :tid THEN away_score END) AS rs10,
            SUM(CASE WHEN home_team_id = :tid THEN away_score
                     WHEN away_team_id = :tid THEN home_score END) AS ra10
        FROM (
            SELECT * FROM games
            WHERE (home_team_id = :tid OR away_team_id = :tid)
              AND game_date < :as_of
              AND status = 'Final'
            ORDER BY game_date DESC LIMIT 10
        ) recent
    """), {"tid": team_id, "as_of": as_of_date}).fetchone()
    w10  = int(last10.w10  or 0)
    rs10 = int(last10.rs10 or 0)
    ra10 = int(last10.ra10 or 0)
    last10_win_rate  = w10 / 10 if w10 is not None else 0.5
    run_diff_last10  = rs10 - ra10

    # 연승/연패
    streak_row = session.execute(text("""
        SELECT
            CASE WHEN home_team_id = :tid AND home_score > away_score THEN 1
                 WHEN away_team_id = :tid AND away_score > home_score THEN 1
                 ELSE -1 END AS result
        FROM games
        WHERE (home_team_id = :tid OR away_team_id = :tid)
          AND game_date < :as_of
          AND status = 'Final'
        ORDER BY game_date DESC LIMIT 10
    """), {"tid": team_id, "as_of": as_of_date}).fetchall()

    streak = 0
    if streak_row:
        sign = streak_row[0].result
        for r in streak_row:
            if r.result == sign:
                streak += sign
            else:
                break

    # 휴식일
    last_game = session.execute(text("""
        SELECT MAX(game_date) AS last_date FROM games
        WHERE (home_team_id = :tid OR away_team_id = :tid)
          AND game_date < :as_of
          AND status = 'Final'
    """), {"tid": team_id, "as_of": as_of_date}).fetchone()
    rest_days = 0
    if last_game and last_game.last_date:
        rest_days = (as_of_date - last_game.last_date).days

    return {
        f"{prefix}_season_win_rate":    season_win_rate,
        f"{prefix}_last10_win_rate":    last10_win_rate,
        f"{prefix}_home_win_rate":      home_win_rate,
        f"{prefix}_streak_signed":      float(streak),
        f"{prefix}_pythagenpat_wp":     pyth_wp,
        f"{prefix}_run_diff_last10":    float(run_diff_last10),
        f"{prefix}_rest_days":          float(rest_days),
    }

"""47개 피처 빌더 — BBref(팀) + Statcast 개인(투수/타자) + Live Feed 라인업/선발/휴식일."""
from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sqlalchemy import text
from sqlalchemy.orm import Session

# v1 모듈에서 BBref CSV 투수 로더와 폴백 기본값 재사용 (batting은 dWAR 포함 v2 로더 사용)
from src.ml.features.bref_features import (
    DATA_RAW,
    LEAGUE_PARK,
    LEAGUE_PITCH,
    _EXCL,
    _norm,
    load_bref_pitching,
    load_park_factors,
)

__all__ = [
    "FEATURE_COLS_V2",
    "FEATURE_COLS_V3",
    "LEAGUE_BAT_V2",
    "LEAGUE_SC_PITCH",
    "LEAGUE_SC_BAT",
    "DEFAULT_REST",
    "build_feature_row_v2",
    "add_diff_features",
    "load_bref_batting_v2",
    "load_pitcher_individual_from_db",
    "load_batter_individual_from_db",
    "load_team_bat_statcast_from_db",
    "get_rest_days",
    "load_bref_pitching",
    "load_park_factors",
]


# v1 LEAGUE_BAT 에 dwar 추가
LEAGUE_BAT_V2 = {
    "ops": 0.726, "obp": 0.320, "slg": 0.410,
    "ba": 0.248, "hr_pa": 0.031, "dwar": 0.0,
}


FEATURE_COLS_V2: list[str] = [
    # 롤링 승률 (3)
    "home_roll_win", "away_roll_win", "roll_win_diff",
    # 팀 투구 (12)
    "home_era", "away_era", "era_diff",
    "home_fip", "away_fip", "fip_diff",
    "home_whip", "away_whip",
    "home_k9", "away_k9",
    "home_bb9", "away_bb9",
    # 선발 투수 (8)
    "home_starter_era", "away_starter_era",
    "home_starter_velo", "away_starter_velo",
    "home_starter_spin", "away_starter_spin",
    "home_starter_whiff", "away_starter_whiff",
    # 팀 타격 (13)
    "home_ops", "away_ops", "ops_diff",
    "home_obp", "away_obp",
    "home_slg", "away_slg",
    "home_ba", "away_ba",
    "home_hr_pa", "away_hr_pa",
    "home_dwar", "away_dwar",
    # 라인업 Statcast (6)
    "home_bat_ev", "away_bat_ev",
    "home_bat_la", "away_bat_la",
    "home_bat_hardhit", "away_bat_hardhit",
    # 휴식일 (2)
    "home_rest", "away_rest",
    # 구장 (3)
    "park_run_factor", "park_hr_factor", "is_dome",
]

assert len(FEATURE_COLS_V2) == 47, f"FEATURE_COLS_V2 must be 47, got {len(FEATURE_COLS_V2)}"


# v3: 47 + difference 피처 7개 = 54개
FEATURE_COLS_V3: list[str] = FEATURE_COLS_V2 + [
    "dwar_diff",          # home_dwar - away_dwar
    "bat_ev_diff",        # home_bat_ev - away_bat_ev
    "bat_hardhit_diff",   # home_bat_hardhit - away_bat_hardhit
    "starter_velo_diff",  # home_starter_velo - away_starter_velo
    "starter_whiff_diff", # home_starter_whiff - away_starter_whiff
    "starter_era_diff",   # away_starter_era - home_starter_era (양수=홈 유리)
    "rest_diff",          # home_rest - away_rest
]

assert len(FEATURE_COLS_V3) == 54, f"FEATURE_COLS_V3 must be 54, got {len(FEATURE_COLS_V3)}"


def add_diff_features(row: dict[str, float]) -> dict[str, float]:
    """47피처 dict → 54피처 (difference 7개 추가)."""
    out = dict(row)
    out["dwar_diff"] = row["home_dwar"] - row["away_dwar"]
    out["bat_ev_diff"] = row["home_bat_ev"] - row["away_bat_ev"]
    out["bat_hardhit_diff"] = row["home_bat_hardhit"] - row["away_bat_hardhit"]
    out["starter_velo_diff"] = row["home_starter_velo"] - row["away_starter_velo"]
    out["starter_whiff_diff"] = row["home_starter_whiff"] - row["away_starter_whiff"]
    out["starter_era_diff"] = row["away_starter_era"] - row["home_starter_era"]
    out["rest_diff"] = row["home_rest"] - row["away_rest"]
    return out


LEAGUE_SC_PITCH = {"velo": 93.0, "spin": 2200.0, "whiff": 0.25}
LEAGUE_SC_BAT = {"ev": 88.5, "la": 12.0, "hard_hit": 0.35}
DEFAULT_REST = 1


# ──────────────────────────────────────────
# BBref batting v2 — dWAR 포함
# ──────────────────────────────────────────


def _load_dwar_master() -> dict[tuple[str, int], float]:
    """bref_dwar_master.csv → {(player_id, season): dWAR}.

    파일이 없으면 빈 dict 반환 (graceful fallback).
    """
    path = Path(DATA_RAW) / "bref_dwar_master.csv"
    if not path.exists():
        return {}
    try:
        df = pd.read_csv(path)
        if "player_id" not in df.columns or "season" not in df.columns or "dWAR" not in df.columns:
            return {}
        return {
            (str(r.player_id), int(r.season)): float(r.dWAR)
            for r in df.itertuples(index=False)
        }
    except Exception:
        return {}


def load_bref_batting_v2(
    seasons: Iterable[int] = (2023, 2024, 2025, 2026),
) -> dict[tuple[str, int], dict]:
    """팀 타격 + dWAR 집계 → {(abbr, season): {...}}.

    dWAR 우선순위:
      1) bref_dwar_master.csv (player_id 기반 join — 가장 정확)
      2) BBref CSV의 dWAR 컬럼 (구버전 호환)
      3) 0.0 폴백
    """
    dwar_master = _load_dwar_master()

    result: dict[tuple[str, int], dict] = {}
    for season in seasons:
        path = Path(DATA_RAW) / f"bref_batting_{season}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if "Rk" in df.columns:
            df = df[pd.notna(df["Rk"])]
        df = df[~df["Team"].astype(str).str.strip().isin(_EXCL)].copy()

        # dWAR 채우기: 마스터 CSV 우선, 없으면 컬럼 → 없으면 0
        if dwar_master and "Player-additional" in df.columns:
            df["dWAR"] = df["Player-additional"].astype(str).str.strip().map(
                lambda pid: dwar_master.get((pid, season), 0.0)
            )
        elif "dWAR" not in df.columns:
            df["dWAR"] = 0.0

        for col in ["PA", "AB", "H", "HR", "BB", "HBP", "SF", "TB", "dWAR"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        df["Team"] = df["Team"].astype(str).str.strip()

        grp = df.groupby("Team").agg(
            total_pa=("PA", "sum"),
            total_ab=("AB", "sum"),
            total_h=("H", "sum"),
            total_hr=("HR", "sum"),
            total_bb=("BB", "sum"),
            total_hbp=("HBP", "sum"),
            total_sf=("SF", "sum"),
            total_tb=("TB", "sum"),
            total_dwar=("dWAR", "sum"),
        ).reset_index()

        for _, row in grp.iterrows():
            abbr = _norm(row["Team"])
            ab = row["total_ab"]
            den = ab + row["total_bb"] + row["total_hbp"] + row["total_sf"]
            if ab < 10 or den < 1:
                continue
            obp = (row["total_h"] + row["total_bb"] + row["total_hbp"]) / den
            slg = row["total_tb"] / ab
            result[(abbr, season)] = {
                "ops": obp + slg,
                "obp": obp,
                "slg": slg,
                "ba": row["total_h"] / ab,
                "hr_pa": row["total_hr"] / row["total_pa"] if row["total_pa"] > 0 else 0.030,
                "dwar": float(row["total_dwar"]),
            }
    return result


# ──────────────────────────────────────────
# DB 기반 Statcast 집계 (학습/추론 공통)
# ──────────────────────────────────────────


def load_pitcher_individual_from_db(session: Session, as_of: date) -> dict[int, dict]:
    """선수별 평균(velo, spin, whiff) — as_of 이전 1.X시즌 데이터만."""
    sql = text("""
        SELECT pitcher_id,
               AVG(release_speed) AS velo,
               AVG(spin_rate)     AS spin,
               AVG(CASE WHEN whiff THEN 1.0 ELSE 0.0 END) AS whiff
        FROM statcast_pitches
        WHERE pitcher_id IS NOT NULL
          AND release_speed IS NOT NULL
          AND game_date < :as_of
          AND game_date >= :since
        GROUP BY pitcher_id
    """)
    since = date(as_of.year - 1, 1, 1)
    rows = session.execute(sql, {"as_of": as_of, "since": since}).fetchall()
    return {
        int(r.pitcher_id): {
            "velo": float(r.velo) if r.velo is not None else LEAGUE_SC_PITCH["velo"],
            "spin": float(r.spin) if r.spin is not None else LEAGUE_SC_PITCH["spin"],
            "whiff": float(r.whiff) if r.whiff is not None else LEAGUE_SC_PITCH["whiff"],
        }
        for r in rows
    }


def load_batter_individual_from_db(
    session: Session,
    batter_ids: Iterable[int],
    as_of: date,
) -> dict[int, dict]:
    """라인업 9명 핀포인트 조회 — as_of 이전 1.X시즌만, BIP(hit_into_play)만."""
    ids = list({int(b) for b in batter_ids if b})
    if not ids:
        return {}
    sql = text("""
        SELECT batter_id,
               AVG(launch_speed) AS ev,
               AVG(launch_angle) AS la,
               AVG(CASE WHEN is_hard_hit THEN 1.0 ELSE 0.0 END) AS hh
        FROM statcast_pitches
        WHERE batter_id = ANY(:ids)
          AND launch_speed IS NOT NULL
          AND description = 'hit_into_play'
          AND game_date < :as_of
          AND game_date >= :since
        GROUP BY batter_id
    """)
    since = date(as_of.year - 1, 1, 1)
    rows = session.execute(sql, {"ids": ids, "as_of": as_of, "since": since}).fetchall()
    return {
        int(r.batter_id): {
            "ev": float(r.ev) if r.ev is not None else LEAGUE_SC_BAT["ev"],
            "la": float(r.la) if r.la is not None else LEAGUE_SC_BAT["la"],
            "hard_hit": float(r.hh) if r.hh is not None else LEAGUE_SC_BAT["hard_hit"],
        }
        for r in rows
    }


def load_team_bat_statcast_from_db(
    session: Session,
    as_of: date,
) -> dict[tuple[str, int], dict]:
    """팀 평균(폴백용) → {(abbr, season): {...}}. BIP(hit_into_play)만."""
    sql = text("""
        SELECT t.abbreviation AS abbr,
               EXTRACT(YEAR FROM sp.game_date)::int AS season,
               AVG(sp.launch_speed) AS ev,
               AVG(sp.launch_angle) AS la,
               AVG(CASE WHEN sp.is_hard_hit THEN 1.0 ELSE 0.0 END) AS hh
        FROM statcast_pitches sp
        JOIN teams t ON t.mlbam_team_id = sp.batter_team_id
        WHERE sp.launch_speed IS NOT NULL
          AND sp.description = 'hit_into_play'
          AND sp.game_date < :as_of
        GROUP BY t.abbreviation, EXTRACT(YEAR FROM sp.game_date)
    """)
    rows = session.execute(sql, {"as_of": as_of}).fetchall()
    return {
        (str(r.abbr), int(r.season)): {
            "ev": float(r.ev) if r.ev is not None else LEAGUE_SC_BAT["ev"],
            "la": float(r.la) if r.la is not None else LEAGUE_SC_BAT["la"],
            "hard_hit": float(r.hh) if r.hh is not None else LEAGUE_SC_BAT["hard_hit"],
        }
        for r in rows
    }


def get_rest_days(session: Session, team_id: int, before: date) -> int:
    """team_id의 직전 Final 경기일 기준 휴식일 — 1~5일로 클램프."""
    sql = text("""
        SELECT MAX(game_date) AS last_dt FROM games
        WHERE (home_team_id = :tid OR away_team_id = :tid)
          AND game_date < :before
          AND status = 'Final'
    """)
    r = session.execute(sql, {"tid": team_id, "before": before}).fetchone()
    if not r or not r.last_dt:
        return DEFAULT_REST
    days = (before - r.last_dt).days
    return max(1, min(days, 5))


# ──────────────────────────────────────────
# 47피처 빌더 (순수 함수 — DB 의존 없음)
# ──────────────────────────────────────────


def _lineup_avg(
    ids: list[int],
    team_abbr: str,
    target_season: int,
    sc_batter_indiv: dict[int, dict],
    sc_team_bat: dict[tuple[str, int], dict],
) -> dict:
    """라인업 9명 평균 → 매칭된 게 하나라도 있으면 그들 평균, 없으면 팀 평균 폴백."""
    rows = [sc_batter_indiv[pid] for pid in ids if pid in sc_batter_indiv]
    if rows:
        return {k: float(np.mean([r[k] for r in rows])) for k in ("ev", "la", "hard_hit")}
    return sc_team_bat.get((team_abbr, target_season), LEAGUE_SC_BAT)


def build_feature_row_v2(
    *,
    h_abbr: str,
    a_abbr: str,
    season: int,
    h_roll: float,
    a_roll: float,
    h_starter_id: int | None,
    a_starter_id: int | None,
    h_lineup_ids: list[int],
    a_lineup_ids: list[int],
    h_team_id: int,
    a_team_id: int,
    pitch_team: dict,
    bat_team: dict,
    park: dict,
    sc_pitcher_indiv: dict[int, dict],
    sc_team_bat: dict[tuple[str, int], dict],
    sc_batter_indiv: dict[int, dict],
    rest_cache: dict[int, int],
) -> dict[str, float]:
    """47개 피처 딕셔너리 반환. 결측은 리그 평균 폴백."""

    # season 폴백 (e.g. 2026 BBref 비어있으면 2025)
    target_season = season if (h_abbr, season) in pitch_team else season - 1

    hp = pitch_team.get((h_abbr, target_season), LEAGUE_PITCH)
    ap = pitch_team.get((a_abbr, target_season), LEAGUE_PITCH)
    hb = bat_team.get((h_abbr, target_season), LEAGUE_BAT_V2)
    ab_ = bat_team.get((a_abbr, target_season), LEAGUE_BAT_V2)
    pk = park.get(h_abbr, LEAGUE_PARK)

    # 선발 투수 개인 ERA — pitcher_game_logs 미연동 시 팀 평균 폴백 (옵션 A)
    h_sp_era = hp["era"]
    a_sp_era = ap["era"]

    # Statcast 투수 개인 (mlbam_id 기반)
    h_sc_p = sc_pitcher_indiv.get(h_starter_id, LEAGUE_SC_PITCH) if h_starter_id else LEAGUE_SC_PITCH
    a_sc_p = sc_pitcher_indiv.get(a_starter_id, LEAGUE_SC_PITCH) if a_starter_id else LEAGUE_SC_PITCH

    # 라인업 9명 평균 → 폴백 팀 평균
    h_sc_b = _lineup_avg(h_lineup_ids, h_abbr, target_season, sc_batter_indiv, sc_team_bat)
    a_sc_b = _lineup_avg(a_lineup_ids, a_abbr, target_season, sc_batter_indiv, sc_team_bat)

    h_rest = rest_cache.get(h_team_id, DEFAULT_REST)
    a_rest = rest_cache.get(a_team_id, DEFAULT_REST)

    return {
        "home_roll_win": float(h_roll),
        "away_roll_win": float(a_roll),
        "roll_win_diff": float(h_roll - a_roll),
        "home_era": float(hp["era"]),
        "away_era": float(ap["era"]),
        "era_diff": float(ap["era"] - hp["era"]),
        "home_fip": float(hp["fip"]),
        "away_fip": float(ap["fip"]),
        "fip_diff": float(ap["fip"] - hp["fip"]),
        "home_whip": float(hp["whip"]),
        "away_whip": float(ap["whip"]),
        "home_k9": float(hp["k9"]),
        "away_k9": float(ap["k9"]),
        "home_bb9": float(hp["bb9"]),
        "away_bb9": float(ap["bb9"]),
        "home_starter_era": float(h_sp_era),
        "away_starter_era": float(a_sp_era),
        "home_starter_velo": float(h_sc_p["velo"]),
        "away_starter_velo": float(a_sc_p["velo"]),
        "home_starter_spin": float(h_sc_p["spin"]),
        "away_starter_spin": float(a_sc_p["spin"]),
        "home_starter_whiff": float(h_sc_p["whiff"]),
        "away_starter_whiff": float(a_sc_p["whiff"]),
        "home_ops": float(hb["ops"]),
        "away_ops": float(ab_["ops"]),
        "ops_diff": float(hb["ops"] - ab_["ops"]),
        "home_obp": float(hb["obp"]),
        "away_obp": float(ab_["obp"]),
        "home_slg": float(hb["slg"]),
        "away_slg": float(ab_["slg"]),
        "home_ba": float(hb["ba"]),
        "away_ba": float(ab_["ba"]),
        "home_hr_pa": float(hb["hr_pa"]),
        "away_hr_pa": float(ab_["hr_pa"]),
        "home_dwar": float(hb.get("dwar", 0.0)),
        "away_dwar": float(ab_.get("dwar", 0.0)),
        "home_bat_ev": float(h_sc_b["ev"]),
        "away_bat_ev": float(a_sc_b["ev"]),
        "home_bat_la": float(h_sc_b["la"]),
        "away_bat_la": float(a_sc_b["la"]),
        "home_bat_hardhit": float(h_sc_b["hard_hit"]),
        "away_bat_hardhit": float(a_sc_b["hard_hit"]),
        "home_rest": float(h_rest),
        "away_rest": float(a_rest),
        "park_run_factor": float(pk.get("run_factor", 1.0)),
        "park_hr_factor": float(pk.get("hr_factor", 1.0)),
        "is_dome": float(bool(pk.get("is_dome", False))),
    }

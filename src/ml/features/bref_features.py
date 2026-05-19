"""BBref CSV → 팀별 집계 피처 (학습/추론 공통 모듈)"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

DATA_RAW = Path("data/raw")

BREF_TO_MLB: dict[str, str] = {
    "CHW": "CWS", "KCR": "KC", "SDP": "SD",
    "SFG": "SF",  "TBR": "TB", "WSN": "WSH",
}
_EXCL = {"TOT", "2TM", "3TM"}

FEATURE_COLS: list[str] = [
    "home_roll_win", "away_roll_win", "roll_win_diff",
    "home_era", "away_era", "era_diff",
    "home_fip", "away_fip", "fip_diff",
    "home_whip", "away_whip",
    "home_k9", "away_k9",
    "home_bb9", "away_bb9",
    "home_ops", "away_ops", "ops_diff",
    "home_obp", "away_obp",
    "home_slg", "away_slg",
    "home_ba", "away_ba",
    "home_hr_pa", "away_hr_pa",
    "park_run_factor", "park_hr_factor", "is_dome",
]

LEAGUE_PITCH = {"era": 4.33, "fip": 4.20, "whip": 1.27, "k9": 8.7, "bb9": 3.0}
LEAGUE_BAT   = {"ops": 0.726, "obp": 0.320, "slg": 0.410, "ba": 0.248, "hr_pa": 0.031}
LEAGUE_PARK  = {"run_factor": 1.0, "hr_factor": 1.0, "is_dome": False}


def _norm(abbr: str) -> str:
    return BREF_TO_MLB.get(str(abbr).strip(), str(abbr).strip())


def load_bref_pitching() -> dict[tuple[str, int], dict]:
    result: dict[tuple[str, int], dict] = {}
    for season in [2023, 2024, 2025]:
        path = DATA_RAW / f"bref_pitching_{season}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        df = df[pd.notna(df["Rk"]) & (~df["Team"].astype(str).str.strip().isin(_EXCL))].copy()
        for col in ["IP", "ER", "H", "BB", "SO", "FIP"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df["Team"] = df["Team"].astype(str).str.strip()

        grp = df.groupby("Team").agg(
            total_ip=("IP",  "sum"),
            total_er=("ER",  "sum"),
            total_h =("H",   "sum"),
            total_bb=("BB",  "sum"),
            total_so=("SO",  "sum"),
            mean_fip=("FIP", "mean"),
        ).reset_index()

        for _, row in grp.iterrows():
            abbr = _norm(row["Team"])
            ip = row["total_ip"]
            if ip < 10:
                continue
            result[(abbr, season)] = {
                "era":  row["total_er"] * 9.0 / ip,
                "fip":  float(row["mean_fip"]) if not np.isnan(row["mean_fip"]) else 4.20,
                "whip": (row["total_h"] + row["total_bb"]) / ip,
                "k9":   row["total_so"] * 9.0 / ip,
                "bb9":  row["total_bb"] * 9.0 / ip,
            }
    return result


def load_bref_batting() -> dict[tuple[str, int], dict]:
    result: dict[tuple[str, int], dict] = {}
    for season in [2023, 2024, 2025]:
        path = DATA_RAW / f"bref_batting_{season}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        df = df[pd.notna(df["Rk"]) & (~df["Team"].astype(str).str.strip().isin(_EXCL))].copy()
        for col in ["PA", "AB", "H", "HR", "BB", "HBP", "SF", "TB"]:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        df["Team"] = df["Team"].astype(str).str.strip()

        grp = df.groupby("Team").agg(
            total_pa =("PA",  "sum"),
            total_ab =("AB",  "sum"),
            total_h  =("H",   "sum"),
            total_hr =("HR",  "sum"),
            total_bb =("BB",  "sum"),
            total_hbp=("HBP", "sum"),
            total_sf =("SF",  "sum"),
            total_tb =("TB",  "sum"),
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
                "ops":   obp + slg,
                "obp":   obp,
                "slg":   slg,
                "ba":    row["total_h"] / ab,
                "hr_pa": row["total_hr"] / row["total_pa"] if row["total_pa"] > 0 else 0.030,
            }
    return result


def load_park_factors() -> dict[str, dict]:
    pj = DATA_RAW / "park_factors.json"
    if pj.exists():
        with open(pj) as f:
            return json.load(f)
    pc = DATA_RAW / "park_factors.csv"
    df = pd.read_csv(pc)
    out: dict[str, dict] = {}
    for _, row in df.iterrows():
        out[str(row["team"])] = {
            "run_factor": float(row["run_factor"]),
            "hr_factor":  float(row["hr_factor"]),
            "is_dome":    str(row.get("is_dome", "False")).lower() == "true",
        }
    return out


def build_feature_row(
    h_abbr: str,
    a_abbr: str,
    season: int,
    h_roll: float,
    a_roll: float,
    pitch_stats: dict,
    bat_stats: dict,
    park_factors: dict,
) -> dict[str, float]:
    """단일 경기 피처 딕셔너리 반환 (FEATURE_COLS 순서와 일치)"""
    prev = season - 1 if season >= 2024 else season

    hp  = pitch_stats.get((h_abbr, prev), LEAGUE_PITCH)
    ap  = pitch_stats.get((a_abbr, prev), LEAGUE_PITCH)
    hb  = bat_stats.get((h_abbr, prev),   LEAGUE_BAT)
    ab_ = bat_stats.get((a_abbr, prev),   LEAGUE_BAT)
    pk  = park_factors.get(h_abbr,         LEAGUE_PARK)

    return {
        "home_roll_win":  h_roll,
        "away_roll_win":  a_roll,
        "roll_win_diff":  h_roll - a_roll,
        "home_era":  hp["era"],  "away_era":  ap["era"],  "era_diff":  ap["era"]  - hp["era"],
        "home_fip":  hp["fip"],  "away_fip":  ap["fip"],  "fip_diff":  ap["fip"]  - hp["fip"],
        "home_whip": hp["whip"], "away_whip": ap["whip"],
        "home_k9":   hp["k9"],   "away_k9":   ap["k9"],
        "home_bb9":  hp["bb9"],  "away_bb9":  ap["bb9"],
        "home_ops":   hb["ops"],   "away_ops":   ab_["ops"],  "ops_diff": hb["ops"] - ab_["ops"],
        "home_obp":   hb["obp"],   "away_obp":   ab_["obp"],
        "home_slg":   hb["slg"],   "away_slg":   ab_["slg"],
        "home_ba":    hb["ba"],    "away_ba":    ab_["ba"],
        "home_hr_pa": hb["hr_pa"], "away_hr_pa": ab_["hr_pa"],
        "park_run_factor": float(pk.get("run_factor", 1.0)),
        "park_hr_factor":  float(pk.get("hr_factor",  1.0)),
        "is_dome":         float(bool(pk.get("is_dome", False))),
    }

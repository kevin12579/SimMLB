"""
MLB 구장별 파크팩터 수집기
============================
소스 우선순위:
  1순위 — Baseball Savant park-factors leaderboard (CSV, 연도별)
  2순위 — pybaseball statcast_batter_expected_stats (xwOBA 등 보조)
  3순위 — 하드코딩 기준값 (2023~2025 평균, 소스 실패 시 폴백)

출력:
  data/raw/park_factors.csv          — 연도별 전체 (long format)
  data/raw/park_factors_latest.csv   — 가장 최근 시즌 기준 (모델 실시간 사용)
  data/raw/park_factors.json         — config/ 연동용 JSON

사용법:
  pip install pybaseball pandas requests
  python park_factor_collector.py
"""

import os
import json
import time
import logging
import requests
import pandas as pd
from io import StringIO
from pathlib import Path
from datetime import date

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── 설정 ──────────────────────────────────────────────────────────────────────

SEASONS    = [2023, 2024, 2025]
OUTPUT_DIR = Path("data/raw")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://baseballsavant.mlb.com",
}

# ── 30개 MLB 팀 구장 메타데이터 ──────────────────────────────────────────────
# venue_id: MLB StatsAPI venue ID
# 돔구장, 해발고도, 좌우 담장 거리 포함
STADIUM_META = {
    # AL East
    "BAL": {"name": "Oriole Park at Camden Yards",    "venue_id": 2,   "is_dome": False, "altitude_ft": 52,   "lf": 333, "cf": 410, "rf": 318},
    "BOS": {"name": "Fenway Park",                     "venue_id": 3,   "is_dome": False, "altitude_ft": 20,   "lf": 310, "cf": 420, "rf": 302},
    "NYY": {"name": "Yankee Stadium",                  "venue_id": 3313,"is_dome": False, "altitude_ft": 55,   "lf": 318, "cf": 408, "rf": 314},
    "TBR": {"name": "Tropicana Field",                 "venue_id": 12,  "is_dome": True,  "altitude_ft": 15,   "lf": 315, "cf": 404, "rf": 322},
    "TOR": {"name": "Rogers Centre",                   "venue_id": 14,  "is_dome": True,  "altitude_ft": 76,   "lf": 328, "cf": 400, "rf": 328},
    # AL Central
    "CHW": {"name": "Guaranteed Rate Field",           "venue_id": 4,   "is_dome": False, "altitude_ft": 595,  "lf": 330, "cf": 400, "rf": 335},
    "CLE": {"name": "Progressive Field",               "venue_id": 5,   "is_dome": False, "altitude_ft": 653,  "lf": 325, "cf": 405, "rf": 325},
    "DET": {"name": "Comerica Park",                   "venue_id": 2394,"is_dome": False, "altitude_ft": 600,  "lf": 345, "cf": 420, "rf": 330},
    "KCR": {"name": "Kauffman Stadium",                "venue_id": 7,   "is_dome": False, "altitude_ft": 750,  "lf": 330, "cf": 410, "rf": 330},
    "MIN": {"name": "Target Field",                    "venue_id": 3312,"is_dome": False, "altitude_ft": 841,  "lf": 339, "cf": 404, "rf": 328},
    # AL West
    "HOU": {"name": "Minute Maid Park",                "venue_id": 2392,"is_dome": False, "altitude_ft": 43,   "lf": 315, "cf": 435, "rf": 326},
    "LAA": {"name": "Angel Stadium",                   "venue_id": 1,   "is_dome": False, "altitude_ft": 160,  "lf": 333, "cf": 400, "rf": 330},
    "OAK": {"name": "Oakland Coliseum",                "venue_id": 10,  "is_dome": False, "altitude_ft": 25,   "lf": 330, "cf": 400, "rf": 330},
    "SEA": {"name": "T-Mobile Park",                   "venue_id": 680, "is_dome": False, "altitude_ft": 0,    "lf": 331, "cf": 401, "rf": 326},
    "TEX": {"name": "Globe Life Field",                "venue_id": 5325,"is_dome": True,  "altitude_ft": 551,  "lf": 329, "cf": 407, "rf": 326},
    # NL East
    "ATL": {"name": "Truist Park",                     "venue_id": 4705,"is_dome": False, "altitude_ft": 1050, "lf": 335, "cf": 400, "rf": 325},
    "MIA": {"name": "loanDepot park",                  "venue_id": 4169,"is_dome": True,  "altitude_ft": 6,    "lf": 344, "cf": 407, "rf": 335},
    "NYM": {"name": "Citi Field",                      "venue_id": 3289,"is_dome": False, "altitude_ft": 20,   "lf": 335, "cf": 408, "rf": 330},
    "PHI": {"name": "Citizens Bank Park",              "venue_id": 2681,"is_dome": False, "altitude_ft": 20,   "lf": 329, "cf": 401, "rf": 330},
    "WSN": {"name": "Nationals Park",                  "venue_id": 3309,"is_dome": False, "altitude_ft": 1,    "lf": 336, "cf": 402, "rf": 335},
    # NL Central
    "CHC": {"name": "Wrigley Field",                   "venue_id": 17,  "is_dome": False, "altitude_ft": 595,  "lf": 355, "cf": 400, "rf": 353},
    "CIN": {"name": "Great American Ball Park",        "venue_id": 2602,"is_dome": False, "altitude_ft": 483,  "lf": 328, "cf": 404, "rf": 325},
    "MIL": {"name": "American Family Field",           "venue_id": 32,  "is_dome": False, "altitude_ft": 635,  "lf": 344, "cf": 400, "rf": 345},
    "PIT": {"name": "PNC Park",                        "venue_id": 31,  "is_dome": False, "altitude_ft": 730,  "lf": 325, "cf": 399, "rf": 320},
    "STL": {"name": "Busch Stadium",                   "venue_id": 2889,"is_dome": False, "altitude_ft": 465,  "lf": 336, "cf": 400, "rf": 335},
    # NL West
    "ARI": {"name": "Chase Field",                     "venue_id": 15,  "is_dome": True,  "altitude_ft": 1082, "lf": 330, "cf": 407, "rf": 334},
    "COL": {"name": "Coors Field",                     "venue_id": 19,  "is_dome": False, "altitude_ft": 5280, "lf": 347, "cf": 415, "rf": 350},
    "LAD": {"name": "Dodger Stadium",                  "venue_id": 22,  "is_dome": False, "altitude_ft": 512,  "lf": 330, "cf": 395, "rf": 330},
    "SDP": {"name": "Petco Park",                      "venue_id": 2680,"is_dome": False, "altitude_ft": 13,   "lf": 336, "cf": 396, "rf": 322},
    "SFG": {"name": "Oracle Park",                     "venue_id": 2395,"is_dome": False, "altitude_ft": 0,    "lf": 339, "cf": 399, "rf": 309},
}

# ── 폴백용 하드코딩 파크팩터 ─────────────────────────────────────────────────
# 출처: FanGraphs 2022~2024 3년 평균 (5년치 평균과 유사)
# run_factor: 1.0 = 리그 평균, >1 = 타자 유리, <1 = 투수 유리
# hr_factor:  홈런 파크팩터
HARDCODED_PF = {
    "ARI": {"run_factor": 1.034, "hr_factor": 1.062},
    "ATL": {"run_factor": 0.991, "hr_factor": 0.980},
    "BAL": {"run_factor": 0.987, "hr_factor": 0.993},
    "BOS": {"run_factor": 1.043, "hr_factor": 0.964},
    "CHC": {"run_factor": 1.007, "hr_factor": 0.992},
    "CHW": {"run_factor": 0.992, "hr_factor": 1.016},
    "CIN": {"run_factor": 1.074, "hr_factor": 1.148},
    "CLE": {"run_factor": 0.948, "hr_factor": 0.881},
    "COL": {"run_factor": 1.186, "hr_factor": 1.224},   # 쿠어스 필드
    "DET": {"run_factor": 0.952, "hr_factor": 0.916},
    "HOU": {"run_factor": 0.981, "hr_factor": 0.946},
    "KCR": {"run_factor": 0.970, "hr_factor": 0.947},
    "LAA": {"run_factor": 0.978, "hr_factor": 0.942},
    "LAD": {"run_factor": 0.975, "hr_factor": 0.957},
    "MIA": {"run_factor": 0.920, "hr_factor": 0.878},   # 투수 친화적
    "MIL": {"run_factor": 0.984, "hr_factor": 1.011},
    "MIN": {"run_factor": 1.013, "hr_factor": 1.073},
    "NYM": {"run_factor": 0.962, "hr_factor": 0.926},
    "NYY": {"run_factor": 1.042, "hr_factor": 1.155},   # 단거리 RF
    "OAK": {"run_factor": 0.952, "hr_factor": 0.879},
    "PHI": {"run_factor": 1.030, "hr_factor": 1.033},
    "PIT": {"run_factor": 0.971, "hr_factor": 0.919},
    "SDP": {"run_factor": 0.938, "hr_factor": 0.887},   # 투수 친화적
    "SEA": {"run_factor": 0.964, "hr_factor": 0.939},
    "SFG": {"run_factor": 0.950, "hr_factor": 0.870},   # 투수 친화적
    "STL": {"run_factor": 0.986, "hr_factor": 0.980},
    "TBR": {"run_factor": 0.977, "hr_factor": 1.001},
    "TEX": {"run_factor": 1.048, "hr_factor": 1.083},
    "TOR": {"run_factor": 1.001, "hr_factor": 1.007},
    "WSN": {"run_factor": 0.986, "hr_factor": 0.969},
}


# ═════════════════════════════════════════════════════════════════════════════
# Source 1 — Baseball Savant Park Factors Leaderboard
# ═════════════════════════════════════════════════════════════════════════════

def fetch_savant_park_factors(season: int) -> pd.DataFrame:
    """
    Baseball Savant 파크팩터 CSV 다운로드
    URL: https://baseballsavant.mlb.com/leaderboard/park-factors
    파라미터: type=year, year={season}, min=100, csv=true
    """
    url = (
        f"https://baseballsavant.mlb.com/leaderboard/park-factors"
        f"?type=year&batSide=&pitchHand=&homeAway=&min=100&csv=true&year={season}"
    )
    log.info(f"  Baseball Savant 파크팩터 수집: {season}시즌")
    try:
        r = requests.get(url, headers=HEADERS, timeout=20)
        if r.status_code != 200:
            log.warning(f"  Baseball Savant {season}: HTTP {r.status_code}")
            return pd.DataFrame()

        df = pd.read_csv(StringIO(r.text))
        if df.empty:
            return pd.DataFrame()

        log.info(f"  ✅ Savant 파크팩터 {season}: {len(df)}팀 수집")
        log.debug(f"  컬럼: {list(df.columns)}")
        return df

    except Exception as e:
        log.warning(f"  Savant 파크팩터 {season} 실패: {e}")
        return pd.DataFrame()


def normalize_savant_pf(df: pd.DataFrame, season: int) -> pd.DataFrame:
    """
    Savant 파크팩터 컬럼 정규화
    Savant 기준: 100 = 리그 평균 → 1.0으로 변환
    """
    col_map = {
        # Savant 컬럼명 → 표준 컬럼명
        "team_abbrev":  "team",
        "team":         "team",
        "venue":        "venue_name",
        "park_factor":  "run_factor_savant",   # 전체 득점
        "hr":           "hr_factor_savant",
        "doubles":      "2b_factor_savant",
        "triples":      "3b_factor_savant",
        "r":            "run_factor_r_savant",
    }

    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

    # Savant 파크팩터: 100 기준 → 1.0 기준으로 변환
    for col in ["run_factor_savant", "hr_factor_savant", "2b_factor_savant",
                "3b_factor_savant", "run_factor_r_savant"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce") / 100.0

    df["season"] = season
    df["source"] = "baseball_savant"
    return df


# ═════════════════════════════════════════════════════════════════════════════
# Source 2 — FanGraphs (pybaseball 경유)
# ═════════════════════════════════════════════════════════════════════════════

def fetch_fangraphs_park_factors(season: int) -> pd.DataFrame:
    """
    FanGraphs 팀 타격 데이터에서 파크팩터 추출
    fg_team_batting_data()의 'Park_Factor' 컬럼 활용
    """
    try:
        import pybaseball as pb
        log.info(f"  FanGraphs 팀 타격 데이터 수집: {season}시즌")

        df = pb.fg_team_batting_data(season, season)

        # 파크팩터 관련 컬럼 탐색
        pf_cols = [c for c in df.columns if any(
            kw in c.lower() for kw in ["park", "factor", "pf"]
        )]
        log.info(f"  FG 파크팩터 관련 컬럼: {pf_cols}")

        if not pf_cols:
            log.warning("  FanGraphs 팀 데이터에 파크팩터 컬럼 없음")
            return pd.DataFrame()

        keep = ["Team"] + pf_cols
        result = df[[c for c in keep if c in df.columns]].copy()
        result["season"] = season
        result["source"] = "fangraphs"
        result.rename(columns={"Team": "team"}, inplace=True)

        log.info(f"  ✅ FanGraphs 파크팩터 {season}: {len(result)}팀")
        return result

    except Exception as e:
        log.warning(f"  FanGraphs 파크팩터 {season} 실패: {e}")
        return pd.DataFrame()


# ═════════════════════════════════════════════════════════════════════════════
# Source 3 — 하드코딩 폴백
# ═════════════════════════════════════════════════════════════════════════════

def build_hardcoded_pf(season: int) -> pd.DataFrame:
    """
    하드코딩 파크팩터 → DataFrame 변환
    Savant/FG 수집 실패 시 폴백으로 사용
    """
    rows = []
    for team, pf in HARDCODED_PF.items():
        meta = STADIUM_META.get(team, {})
        rows.append({
            "team":          team,
            "season":        season,
            "run_factor":    pf["run_factor"],
            "hr_factor":     pf["hr_factor"],
            "venue_name":    meta.get("name", ""),
            "venue_id":      meta.get("venue_id"),
            "is_dome":       meta.get("is_dome", False),
            "altitude_ft":   meta.get("altitude_ft", 0),
            "lf_ft":         meta.get("lf", 0),
            "cf_ft":         meta.get("cf", 0),
            "rf_ft":         meta.get("rf", 0),
            "source":        "hardcoded",
        })
    return pd.DataFrame(rows)


# ═════════════════════════════════════════════════════════════════════════════
# 통합 수집기
# ═════════════════════════════════════════════════════════════════════════════

def collect_park_factors(seasons: list[int]) -> pd.DataFrame:
    """
    전 소스 순서대로 시도하여 파크팩터 수집
    성공한 소스의 데이터를 하드코딩과 병합하여 완전한 30팀 데이터 생성
    """
    all_dfs = []

    for season in seasons:
        log.info(f"\n{'='*50}")
        log.info(f"시즌: {season}")
        log.info(f"{'='*50}")

        season_df = pd.DataFrame()

        # 1순위: Baseball Savant
        savant_df = fetch_savant_park_factors(season)
        if not savant_df.empty:
            savant_df = normalize_savant_pf(savant_df, season)
            season_df = savant_df
            time.sleep(2.0)

        # 2순위: FanGraphs (Savant 실패 또는 보완)
        fg_df = fetch_fangraphs_park_factors(season)
        if not fg_df.empty:
            if season_df.empty:
                season_df = fg_df
            else:
                # Savant 데이터에 FG 파크팩터 병합
                if "team" in fg_df.columns and "team" in season_df.columns:
                    season_df = season_df.merge(
                        fg_df.drop(columns=["season","source"], errors="ignore"),
                        on="team", how="left"
                    )
            time.sleep(2.0)

        # 3순위: 하드코딩 폴백 (항상 실행 — 누락 팀 보완)
        hc_df = build_hardcoded_pf(season)

        if season_df.empty:
            log.warning(f"  외부 소스 모두 실패 → 하드코딩 사용: {season}")
            season_df = hc_df
        else:
            # 외부 소스에서 누락된 팀을 하드코딩으로 보완
            if "team" in season_df.columns:
                existing_teams = set(season_df["team"].str.upper())
                missing = hc_df[~hc_df["team"].isin(existing_teams)]
                if not missing.empty:
                    log.info(f"  누락 팀 {len(missing)}개 하드코딩으로 보완: {list(missing['team'])}")
                    season_df = pd.concat([season_df, missing], ignore_index=True)

            # 구장 메타데이터 병합 (외부 소스에 없는 경우)
            for col in ["venue_name", "venue_id", "is_dome", "altitude_ft", "lf_ft", "cf_ft", "rf_ft"]:
                if col not in season_df.columns:
                    season_df = season_df.merge(
                        hc_df[["team", col]], on="team", how="left"
                    )

            # run_factor / hr_factor 컬럼 통일
            if "run_factor" not in season_df.columns:
                if "run_factor_savant" in season_df.columns:
                    season_df["run_factor"] = season_df["run_factor_savant"]
                else:
                    season_df = season_df.merge(
                        hc_df[["team", "run_factor", "hr_factor"]],
                        on="team", how="left"
                    )
            if "hr_factor" not in season_df.columns and "hr_factor_savant" in season_df.columns:
                season_df["hr_factor"] = season_df["hr_factor_savant"]

        all_dfs.append(season_df)
        log.info(f"  시즌 {season} 완료: {len(season_df)}팀")

    combined = pd.concat(all_dfs, ignore_index=True)
    return combined


# ═════════════════════════════════════════════════════════════════════════════
# 저장 & JSON 변환
# ═════════════════════════════════════════════════════════════════════════════

def save_outputs(df: pd.DataFrame):
    """
    수집된 파크팩터를 3가지 형식으로 저장
    1. data/raw/park_factors.csv         — 전체 (연도별 long format)
    2. data/raw/park_factors_latest.csv  — 최신 시즌만
    3. data/raw/park_factors.json        — config/ 연동용
    """

    # ── CSV 전체 저장 ──
    csv_path = OUTPUT_DIR / "park_factors.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    log.info(f"\n저장: {csv_path} ({len(df)}행)")

    # ── 최신 시즌 CSV ──
    latest_season = df["season"].max()
    latest_df = df[df["season"] == latest_season].copy()
    latest_path = OUTPUT_DIR / "park_factors_latest.csv"
    latest_df.to_csv(latest_path, index=False, encoding="utf-8-sig")
    log.info(f"저장: {latest_path} ({len(latest_df)}팀, {latest_season}시즌)")

    # ── JSON 저장 (config/ 연동용) ──
    # 구조: { "LAD": { "run_factor": 0.975, "hr_factor": 0.957, "is_dome": false, ... }, ... }
    json_path = OUTPUT_DIR / "park_factors.json"

    json_data = {}
    for _, row in latest_df.iterrows():
        team = str(row.get("team", "")).upper().strip()
        if not team:
            continue
        json_data[team] = {
            "run_factor":   round(float(row.get("run_factor",   1.0) or 1.0), 4),
            "hr_factor":    round(float(row.get("hr_factor",    1.0) or 1.0), 4),
            "venue_name":   str(row.get("venue_name", "")),
            "venue_id":     int(row.get("venue_id",   0) or 0),
            "is_dome":      bool(row.get("is_dome",   False)),
            "altitude_ft":  int(row.get("altitude_ft", 0) or 0),
            "lf_ft":        int(row.get("lf_ft", 0) or 0),
            "cf_ft":        int(row.get("cf_ft", 0) or 0),
            "rf_ft":        int(row.get("rf_ft", 0) or 0),
            "season":       int(latest_season),
            "source":       str(row.get("source", "hardcoded")),
        }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)
    log.info(f"저장: {json_path} ({len(json_data)}팀)")

    # ── 결과 요약 출력 ──
    print_summary(latest_df)
    return csv_path, latest_path, json_path


def print_summary(df: pd.DataFrame):
    """파크팩터 요약 출력"""
    print("\n" + "="*60)
    print(f"구장별 파크팩터 ({df['season'].iloc[0]}시즌)")
    print("="*60)

    # run_factor 기준 정렬
    if "run_factor" in df.columns and "team" in df.columns:
        show = df[["team", "run_factor", "hr_factor", "is_dome", "altitude_ft"]].copy()
        show["run_factor"] = show["run_factor"].round(3)
        show["hr_factor"]  = show["hr_factor"].round(3)
        show = show.sort_values("run_factor", ascending=False)

        print(show.to_string(index=False))

        print("\n🏟️  타자 친화적 구장 TOP 5 (run_factor 높은 순):")
        top5 = show.nlargest(5, "run_factor")
        for _, r in top5.iterrows():
            meta = STADIUM_META.get(r["team"], {})
            print(f"  {r['team']:4s} {meta.get('name',''):<35s} run={r['run_factor']:.3f}  hr={r['hr_factor']:.3f}")

        print("\n⚾  투수 친화적 구장 TOP 5 (run_factor 낮은 순):")
        bot5 = show.nsmallest(5, "run_factor")
        for _, r in bot5.iterrows():
            meta = STADIUM_META.get(r["team"], {})
            print(f"  {r['team']:4s} {meta.get('name',''):<35s} run={r['run_factor']:.3f}  hr={r['hr_factor']:.3f}")

        dome_teams = show[show["is_dome"] == True]["team"].tolist()
        print(f"\n🏠 돔구장 ({len(dome_teams)}개): {', '.join(dome_teams)}")

        coors = show[show["team"] == "COL"]
        if not coors.empty:
            print(f"\n⛰️  쿠어스 필드 (해발 5,280ft): run={coors.iloc[0]['run_factor']:.3f}  (리그 최고 타자 구장)")
    print("="*60)


# ═════════════════════════════════════════════════════════════════════════════
# 메인 실행
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    log.info("MLB 파크팩터 수집 시작")
    log.info(f"대상 시즌: {SEASONS}")
    log.info(f"출력 디렉토리: {OUTPUT_DIR.absolute()}")

    df = collect_park_factors(SEASONS)

    if df.empty:
        log.error("수집 실패 — 모든 소스에서 데이터를 가져오지 못했습니다")
    else:
        csv_path, latest_path, json_path = save_outputs(df)

        print(f"\n✅ 완료!")
        print(f"  전체 CSV:   {csv_path}")
        print(f"  최신 CSV:   {latest_path}")
        print(f"  JSON:       {json_path}")
        print(f"\n사용 예시 (feature_engineering.py):")
        print(f"  import json")
        print(f"  PARK_FACTORS = json.loads(open('{json_path}').read())")
        print(f"  pf = PARK_FACTORS.get('LAD', {{'run_factor': 1.0}})")
        print(f"  adjusted_era = era / pf['run_factor']")
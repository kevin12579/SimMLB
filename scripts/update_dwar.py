"""BBref WAR Daily 마스터 파일 → dWAR 백필 CSV.

데이터 소스:
  https://www.baseball-reference.com/data/war_daily_bat.txt
  https://www.baseball-reference.com/data/war_daily_pit.txt

이 두 파일은 player_ID × year_ID 단위로 매일 업데이트됨.
타자/투수 모두 'WAR_def' 컬럼 = 수비 WAR (= dWAR).

출력:
  data/raw/bref_dwar_master.csv  (player_id, season, dWAR, total_WAR)

주의 (sandbox 차단):
  BBref이 Cloudflare 챌린지를 발동하면 403 또는 챌린지 HTML이 옴.
  이 스크립트는 graceful fail — 기존 마스터 CSV가 있으면 유지.
  사용자 PC에서 실행하면 90%+ 성공.
"""
from __future__ import annotations

import io
import os
import sys
from pathlib import Path

import pandas as pd
import requests

DATA_RAW = Path("data/raw")
DATA_RAW.mkdir(parents=True, exist_ok=True)
OUTPUT = DATA_RAW / "bref_dwar_master.csv"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36"
    ),
    "Accept": "text/plain,text/csv,*/*",
    "Accept-Language": "en-US,en;q=0.5",
}


def _fetch_war(url: str) -> pd.DataFrame:
    """단일 WAR 파일 다운로드 → DataFrame. 차단/오류 시 빈 DataFrame."""
    try:
        r = requests.get(url, headers=HEADERS, timeout=30)
    except Exception as e:
        print(f"  ❌ 네트워크 오류: {e}")
        return pd.DataFrame()

    if r.status_code != 200:
        print(f"  ❌ HTTP {r.status_code} (Cloudflare 차단 가능성)")
        return pd.DataFrame()

    # Cloudflare 챌린지 HTML 감지 (txt가 HTML로 오는 경우)
    if "<!DOCTYPE html>" in r.text[:200] or "Just a moment" in r.text[:200]:
        print("  ❌ Cloudflare 챌린지 페이지 수신 (User-Agent로 우회 실패)")
        return pd.DataFrame()

    try:
        df = pd.read_csv(io.StringIO(r.text), low_memory=False)
    except Exception as e:
        print(f"  ❌ CSV 파싱 오류: {e}")
        return pd.DataFrame()

    df.columns = df.columns.str.strip()
    return df


def _process(df: pd.DataFrame, seasons: list[int]) -> pd.DataFrame:
    """필터 + dWAR 컬럼 정규화 + player×season groupby (멀티팀 합산)."""
    if df.empty or "year_ID" not in df.columns or "player_ID" not in df.columns:
        return pd.DataFrame(columns=["player_id", "season", "dWAR", "total_WAR"])

    # WAR_def: 타자/투수 모두 동일한 컬럼명 사용
    if "WAR_def" not in df.columns:
        df["WAR_def"] = 0.0

    # 안전 변환
    df["year_ID"] = pd.to_numeric(df["year_ID"], errors="coerce")
    df = df.dropna(subset=["year_ID"]).copy()
    df["year_ID"] = df["year_ID"].astype(int)
    df["WAR_def"] = pd.to_numeric(df["WAR_def"], errors="coerce").fillna(0.0)
    df["WAR"] = pd.to_numeric(df.get("WAR", 0.0), errors="coerce").fillna(0.0)

    df = df[df["year_ID"].isin(seasons)].copy()
    if df.empty:
        return pd.DataFrame(columns=["player_id", "season", "dWAR", "total_WAR"])

    # 한 선수가 한 시즌 여러 팀에서 뛴 경우 합산
    grouped = df.groupby(["player_ID", "year_ID"]).agg(
        dWAR=("WAR_def", "sum"),
        total_WAR=("WAR", "sum"),
    ).reset_index()
    grouped = grouped.rename(columns={"player_ID": "player_id", "year_ID": "season"})
    return grouped


def update_dwar(seasons: list[int] | None = None) -> int:
    """타자/투수 WAR 마스터 → bref_dwar_master.csv. 저장된 row 수 반환 (0이면 실패)."""
    seasons = seasons or [2023, 2024, 2025, 2026]
    print(f"BBref WAR Daily → dWAR 백필 (시즌: {seasons})")

    print("[1/2] 타자 WAR 다운로드...")
    bat = _fetch_war("https://www.baseball-reference.com/data/war_daily_bat.txt")
    bat_dwar = _process(bat, seasons)
    print(f"  타자 rows: {len(bat_dwar)}")

    print("[2/2] 투수 WAR 다운로드...")
    pit = _fetch_war("https://www.baseball-reference.com/data/war_daily_pit.txt")
    pit_dwar = _process(pit, seasons)
    print(f"  투수 rows: {len(pit_dwar)}")

    combined = pd.concat([bat_dwar, pit_dwar], ignore_index=True)
    if combined.empty:
        print("\n⚠️ 다운로드 실패 — Cloudflare 차단 환경. 사용자 PC에서 재실행 권장.")
        if OUTPUT.exists():
            print(f"   기존 파일 유지: {OUTPUT}")
        return 0

    # 투타겸업/중복 → player×season 단위로 최종 합산
    final = combined.groupby(["player_id", "season"]).agg(
        dWAR=("dWAR", "sum"),
        total_WAR=("total_WAR", "sum"),
    ).reset_index()

    final.to_csv(OUTPUT, index=False)
    print(f"\n✅ 저장: {OUTPUT} ({len(final)} rows)")
    print(final.head().to_string(index=False))
    return len(final)


def manual_import_from_local(
    bat_path: str | Path,
    pit_path: str | Path | None = None,
    seasons: list[int] | None = None,
) -> int:
    """사용자 PC에서 직접 다운로드한 raw 파일을 import 하는 폴백.

    Args:
        bat_path: war_daily_bat.txt 로컬 경로
        pit_path: war_daily_pit(ch).txt 로컬 경로 (선택, 있으면 합산)
        seasons: 필터할 시즌 리스트
    """
    seasons = seasons or [2023, 2024, 2025, 2026]
    frames: list[pd.DataFrame] = []

    for label, path in [("타자", bat_path), ("투수", pit_path)]:
        if path is None:
            continue
        p = Path(path)
        if not p.exists():
            print(f"⚠️ {label} 파일 없음: {p} — 스킵")
            continue
        print(f"[{label}] 파싱 중: {p}")
        df = pd.read_csv(p, low_memory=False)
        df.columns = df.columns.str.strip()
        processed = _process(df, seasons)
        print(f"  {label} rows: {len(processed)}")
        frames.append(processed)

    if not frames:
        print("❌ import 가능한 파일 없음")
        return 0

    combined = pd.concat(frames, ignore_index=True)
    # 투타겸업/중복 → player×season 단위 합산
    final = combined.groupby(["player_id", "season"]).agg(
        dWAR=("dWAR", "sum"),
        total_WAR=("total_WAR", "sum"),
    ).reset_index()

    final.to_csv(OUTPUT, index=False)
    print(f"\n✅ 로컬 import 완료: {OUTPUT} ({len(final)} rows)")
    print(final.head().to_string(index=False))
    return len(final)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seasons", type=int, nargs="+",
                        default=[2023, 2024, 2025, 2026])
    parser.add_argument("--import-bat", type=str, default=None,
                        help="로컬 war_daily_bat.txt 경로 (Cloudflare 차단 시)")
    parser.add_argument("--import-pit", type=str, default=None,
                        help="로컬 war_daily_pit.txt 경로 (선택)")
    args = parser.parse_args()
    if args.import_bat or args.import_pit:
        n = manual_import_from_local(args.import_bat, args.import_pit, args.seasons)
    else:
        n = update_dwar(args.seasons)
    sys.exit(0 if n > 0 else 1)

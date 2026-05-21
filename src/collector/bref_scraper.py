"""Baseball Reference 스텔스 스크래퍼 — pitching/batting 시즌 누적 덮어쓰기."""
from __future__ import annotations

import random
import time
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

from src.common.logger import get_logger

logger = get_logger(__name__)

DATA_RAW = Path("data/raw")

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/121.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 "
    "(KHTML, like Gecko) Version/17.0 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36",
]

BASE_HEADERS = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Connection": "keep-alive",
}

RATE_LIMIT_SLEEP_SEC = 60
MANNER_SLEEP_SEC = 3
RETRY_SLEEP_SEC = 5


def safe_bref_scrape(url: str, max_retries: int = 3) -> pd.DataFrame:
    """팀원 코드 패턴: 429 → 60s, 매너 대기 3s, UA 로테이션."""
    for attempt in range(max_retries):
        headers = {**BASE_HEADERS, "User-Agent": random.choice(USER_AGENTS)}
        try:
            r = requests.get(url, headers=headers, timeout=15)
            if r.status_code == 429:
                logger.warning(
                    "BBref 429 — %ds 대기 후 재시도 (%d/%d) %s",
                    RATE_LIMIT_SLEEP_SEC, attempt + 1, max_retries, url,
                )
                time.sleep(RATE_LIMIT_SLEEP_SEC)
                continue
            r.raise_for_status()
            tables = pd.read_html(StringIO(r.text))
            time.sleep(MANNER_SLEEP_SEC)
            return tables[0]
        except Exception as e:
            logger.warning("BBref 시도 %d/%d 실패: %s", attempt + 1, max_retries, e)
            time.sleep(RETRY_SLEEP_SEC)
    logger.error("BBref 스크래핑 최종 실패: %s", url)
    return pd.DataFrame()


def update_bref_season(season: int, raw_dir: Path | None = None) -> dict[str, int]:
    """pitching / batting 시즌 누적 → CSV 통째로 덮어쓰기 (mode='w').

    BBref 시즌 누적 평균 (예: 팀 ERA)은 매일 갱신되므로 append 가 아닌 overwrite.
    """
    raw = raw_dir or DATA_RAW
    raw.mkdir(parents=True, exist_ok=True)

    urls = {
        "pitching": f"https://www.baseball-reference.com/leagues/majors/{season}-standard-pitching.shtml",
        "batting":  f"https://www.baseball-reference.com/leagues/majors/{season}-standard-batting.shtml",
    }
    counts: dict[str, int] = {}
    for category, url in urls.items():
        logger.info("BBref %s %s 스크래핑 시작: %s", season, category, url)
        df = safe_bref_scrape(url)
        if df.empty:
            logger.error("BBref %s %s 빈 응답 — CSV 갱신 스킵 (기존 파일 유지)", season, category)
            counts[category] = 0
            continue
        out = raw / f"bref_{category}_{season}.csv"
        df.to_csv(out, index=False)
        counts[category] = len(df)
        logger.info("BBref %s %s 저장: %d rows → %s", season, category, len(df), out)
    return counts


if __name__ == "__main__":
    import argparse
    from datetime import date

    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, default=date.today().year)
    args = parser.parse_args()
    result = update_bref_season(args.season)
    print(f"완료: {result}")

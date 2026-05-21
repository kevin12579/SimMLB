"""game_lineups 과거 백필 — MLB Stats API boxscore.battingOrder.

각 경기의 박스스코어에서 양 팀 battingOrder (1~9번 타자 mlbam_id)를 가져와
game_lineups 테이블에 INSERT.

배치 처리:
  - 동시 요청 3개 (BaseCollector semaphore 활용)
  - 매 100경기마다 진행 로그
  - 이미 백필된 game_pk는 스킵
"""
from __future__ import annotations

import asyncio
import io
import sys
from datetime import date

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.path.insert(0, ".")

from sqlalchemy import text  # noqa: E402

from src.collector.base import BaseCollector  # noqa: E402
from src.common.logger import get_logger  # noqa: E402
# ORM 메타 등록 (FK 해결을 위해 명시 import)
from src.db.models import teams as _teams  # noqa: F401, E402
from src.db.models import players as _players  # noqa: F401, E402
from src.db.models.games import GameLineup  # noqa: E402
from src.db.player_utils import ensure_players_exist  # noqa: E402
from src.db.session import get_session  # noqa: E402

logger = get_logger("backfill_lineups")


class BoxscoreClient(BaseCollector):
    BASE = "https://statsapi.mlb.com/api/v1"

    async def fetch_boxscore(self, game_pk: int) -> dict:
        data = await self._get(f"{self.BASE}/game/{game_pk}/boxscore", timeout=15)
        assert isinstance(data, dict)
        return data


def _extract_lineups(box: dict) -> tuple[list[int], list[int]]:
    """boxscore → (home_lineup, away_lineup) batting_order 순서."""
    teams = box.get("teams", {}) or {}
    home = teams.get("home", {}) or {}
    away = teams.get("away", {}) or {}
    h_order = [int(x) for x in (home.get("battingOrder") or []) if x]
    a_order = [int(x) for x in (away.get("battingOrder") or []) if x]
    return h_order, a_order


async def backfill_game(client: BoxscoreClient, game_pk: int, game_date: date,
                        h_team: int, a_team: int) -> int:
    """한 경기 라인업 INSERT. 반환=삽입 행 수."""
    try:
        box = await client.fetch_boxscore(game_pk)
    except Exception as e:
        logger.warning("game %d boxscore fetch failed: %s", game_pk, e)
        return 0

    h_order, a_order = _extract_lineups(box)
    if not h_order and not a_order:
        return 0

    # FK 위반 방지
    pids = set(h_order + a_order)
    with get_session() as session:
        ensure_players_exist(pids, session)

        # 기존 라인업이 있으면 스킵 (idempotent)
        existing = session.execute(
            text("SELECT 1 FROM game_lineups WHERE game_pk = :pk LIMIT 1"),
            {"pk": game_pk},
        ).fetchone()
        if existing:
            return 0

        inserted = 0
        for is_home, lineup in [(True, h_order), (False, a_order)]:
            team_id = h_team if is_home else a_team
            for order, pid in enumerate(lineup, start=1):
                session.add(GameLineup(
                    game_pk=game_pk, game_date=game_date,
                    player_id=pid, team_id=team_id,
                    batting_order=order, is_home=is_home,
                ))
                inserted += 1
        session.commit()
        return inserted


async def main(seasons: list[int] | None = None, limit: int | None = None) -> None:
    seasons = seasons or [2023, 2024]
    with get_session() as session:
        rows = session.execute(text("""
            SELECT g.game_pk, g.game_date, g.home_team_id, g.away_team_id
            FROM games g
            WHERE g.season = ANY(:seasons) AND g.status='Final'
            AND NOT EXISTS (SELECT 1 FROM game_lineups gl WHERE gl.game_pk = g.game_pk)
            ORDER BY g.game_date
        """), {"seasons": seasons}).fetchall()

    if limit:
        rows = rows[:limit]
    logger.info("백필 대상: %d 경기", len(rows))

    if not rows:
        logger.info("이미 모두 백필됨")
        return

    total_inserted = 0
    async with BoxscoreClient() as client:
        for i, r in enumerate(rows, 1):
            n = await backfill_game(client, int(r.game_pk), r.game_date,
                                     int(r.home_team_id), int(r.away_team_id))
            total_inserted += n
            if i % 100 == 0:
                logger.info("진행 %d/%d  누적 %d rows", i, len(rows), total_inserted)

    logger.info("백필 완료: %d rows", total_inserted)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--seasons", type=int, nargs="+", default=[2023, 2024])
    p.add_argument("--limit", type=int, default=None, help="테스트용 N경기만")
    a = p.parse_args()
    asyncio.run(main(a.seasons, a.limit))

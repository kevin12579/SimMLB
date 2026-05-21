"""MLB StatsAPI v1.1 Live Feed — 한 번의 호출로 날씨+라인업+선발 통합 수집"""
from __future__ import annotations

from typing import TypedDict

from sqlalchemy.orm import Session
from sqlalchemy.dialects.postgresql import insert

from src.collector.base import BaseCollector
from src.db.models.games import Game, GameWeather, GameLineup
from src.db.player_utils import ensure_players_exist
from src.common.logger import get_logger

logger = get_logger(__name__)


class LiveFeedSnapshot(TypedDict):
    game_pk: int
    weather_temp_f: float | None
    weather_condition: str
    weather_wind: str
    home_starter_id: int | None
    away_starter_id: int | None
    home_lineup_ids: list[int]
    away_lineup_ids: list[int]


def _parse_temp(raw: object) -> float | None:
    if raw is None:
        return None
    s = str(raw).strip()
    if not s or s in ("--", "N/A"):
        return None
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


class LiveFeedClient(BaseCollector):
    """단일 경기 Live Feed 통합 수집기."""

    BASE = "https://statsapi.mlb.com/api/v1.1"

    async def fetch_live(self, game_pk: int) -> dict:
        data = await self._get(f"{self.BASE}/game/{game_pk}/feed/live", timeout=12)
        assert isinstance(data, dict)
        return data

    def parse_snapshot(self, game_pk: int, live: dict) -> LiveFeedSnapshot:
        """순수 함수: HTTP 응답 dict → LiveFeedSnapshot. 테스트에서 mock 응답으로 호출."""
        gd = live.get("gameData", {}) or {}
        ld = live.get("liveData", {}) or {}

        weather = gd.get("weather", {}) or {}
        prob = gd.get("probablePitchers", {}) or {}
        boxteams = (ld.get("boxscore", {}) or {}).get("teams", {}) or {}

        h_starter = (prob.get("home") or {}).get("id")
        a_starter = (prob.get("away") or {}).get("id")

        return {
            "game_pk": game_pk,
            "weather_temp_f": _parse_temp(weather.get("temp")),
            "weather_condition": str(weather.get("condition", ""))[:50],
            "weather_wind": str(weather.get("wind", ""))[:50],
            "home_starter_id": int(h_starter) if h_starter else None,
            "away_starter_id": int(a_starter) if a_starter else None,
            "home_lineup_ids": [int(pid) for pid in
                                (boxteams.get("home", {}) or {}).get("battingOrder", []) or []],
            "away_lineup_ids": [int(pid) for pid in
                                (boxteams.get("away", {}) or {}).get("battingOrder", []) or []],
        }

    async def extract_snapshot(self, game_pk: int) -> LiveFeedSnapshot:
        live = await self.fetch_live(game_pk)
        return self.parse_snapshot(game_pk, live)

    async def sync_to_db(self, game_pk: int, session: Session) -> LiveFeedSnapshot:
        """Live Feed → games/game_weather/game_lineups UPSERT."""
        snap = await self.extract_snapshot(game_pk)

        # 1) 선발/라인업 선수 stub insert (FK 위반 방지)
        pids: set[int] = set()
        if snap["home_starter_id"]:
            pids.add(snap["home_starter_id"])
        if snap["away_starter_id"]:
            pids.add(snap["away_starter_id"])
        pids.update(snap["home_lineup_ids"])
        pids.update(snap["away_lineup_ids"])
        if pids:
            ensure_players_exist(pids, session)

        # 2) games 선발 투수 업데이트
        if snap["home_starter_id"] or snap["away_starter_id"]:
            updates: dict = {}
            if snap["home_starter_id"]:
                updates["home_starter_id"] = snap["home_starter_id"]
            if snap["away_starter_id"]:
                updates["away_starter_id"] = snap["away_starter_id"]
            session.query(Game).filter(Game.game_pk == game_pk).update(updates)

        # 3) 날씨 UPSERT
        if snap["weather_temp_f"] is not None:
            stmt = insert(GameWeather).values(
                game_pk=game_pk,
                temp_f=snap["weather_temp_f"],
            ).on_conflict_do_update(
                index_elements=["game_pk"],
                set_={"temp_f": snap["weather_temp_f"]},
            )
            session.execute(stmt)

        # 4) 라인업 DELETE + INSERT (라인업 변경 빈번)
        game = session.query(Game).filter(Game.game_pk == game_pk).first()
        if game:
            session.query(GameLineup).filter(GameLineup.game_pk == game_pk).delete()
            for is_home, lineup in [
                (True, snap["home_lineup_ids"]),
                (False, snap["away_lineup_ids"]),
            ]:
                team_id = game.home_team_id if is_home else game.away_team_id
                for order, pid in enumerate(lineup, start=1):
                    session.add(GameLineup(
                        game_pk=game_pk,
                        game_date=game.game_date,
                        player_id=pid,
                        team_id=team_id,
                        batting_order=order,
                        is_home=is_home,
                    ))

        session.commit()
        logger.info(
            "Live feed synced game %d (home lineup=%d, away lineup=%d, temp=%s)",
            game_pk,
            len(snap["home_lineup_ids"]),
            len(snap["away_lineup_ids"]),
            snap["weather_temp_f"],
        )
        return snap

import asyncio
from datetime import date, datetime
from typing import Any

from sqlalchemy.orm import Session
from sqlalchemy.dialects.postgresql import insert

from src.collector.base import BaseCollector
from src.db.models.teams import Team
from src.db.models.players import Player
from src.db.models.games import Game, TeamDailySnapshot
from src.common.logger import get_logger
from src.db.player_utils import ensure_players_exist

logger = get_logger(__name__)

BASE = "https://statsapi.mlb.com/api/v1"


class MLBStatsAPIClient(BaseCollector):
    BASE_URL = BASE

    # ──────────────────────────────────────────
    # 경기 일정/결과
    # ──────────────────────────────────────────

    async def fetch_schedule(self, game_date: date) -> list[dict]:
        """당일 경기 일정 (game_pk, 홈/원정 팀, 시간, 장소)"""
        data = await self._get(
            f"{BASE}/schedule",
            params={
                "sportId": 1,
                "date": game_date.strftime("%Y-%m-%d"),
                "hydrate": "team,venue,probablePitcher",
            },
        )
        games = []
        for date_entry in data.get("dates", []):
            for g in date_entry.get("games", []):
                games.append(g)
        return games

    async def fetch_game_result(self, game_pk: int) -> dict:
        """경기 결과 (점수, 선발 투수)"""
        data = await self._get(
            f"{BASE}/game/{game_pk}/linescore",
            params={"hydrate": "team"},
        )
        return data

    async def fetch_boxscore(self, game_pk: int) -> dict:
        """박스스코어 (선수별 기록)"""
        return await self._get(f"{BASE}/game/{game_pk}/boxscore")

    # ──────────────────────────────────────────
    # 팀 / 선수
    # ──────────────────────────────────────────

    async def fetch_all_teams(self) -> list[dict]:
        """MLB 30개 팀 목록"""
        data = await self._get(f"{BASE}/teams", params={"sportId": 1})
        return data.get("teams", [])

    async def fetch_roster(self, team_id: int, season: int) -> list[dict]:
        """팀 로스터 (40인)"""
        data = await self._get(
            f"{BASE}/teams/{team_id}/roster",
            params={"rosterType": "40Man", "season": season},
        )
        return data.get("roster", [])

    async def fetch_player_info(self, mlbam_id: int) -> dict:
        """선수 기본 정보"""
        data = await self._get(f"{BASE}/people/{mlbam_id}")
        people = data.get("people", [])
        return people[0] if people else {}

    async def fetch_player_season_stats(self, mlbam_id: int, season: int, group: str) -> dict:
        """선수 시즌 통계 (group: 'pitching' or 'hitting')"""
        data = await self._get(
            f"{BASE}/people/{mlbam_id}/stats",
            params={"stats": "season", "group": group, "season": season},
        )
        stats_list = data.get("stats", [])
        if stats_list and stats_list[0].get("splits"):
            return stats_list[0]["splits"][0].get("stat", {})
        return {}

    # ──────────────────────────────────────────
    # DB UPSERT 헬퍼
    # ──────────────────────────────────────────

    async def sync_teams(self, session: Session) -> None:
        """MLB 30개 팀 정보를 DB에 UPSERT"""
        teams = await self.fetch_all_teams()
        for t in teams:
            venue = t.get("venue", {})
            stmt = insert(Team).values(
                mlbam_team_id=t["id"],
                name=t["name"],
                abbreviation=t.get("abbreviation", ""),
                league=t.get("league", {}).get("name", "")[:5],
                division=t.get("division", {}).get("name", "")[:20],
                venue_id=venue.get("id", 0),
                venue_name=venue.get("name", "")[:100],
            ).on_conflict_do_update(
                index_elements=["mlbam_team_id"],
                set_=dict(
                    name=t["name"],
                    abbreviation=t.get("abbreviation", ""),
                    venue_name=venue.get("name", "")[:100],
                ),
            )
            session.execute(stmt)
        session.commit()
        logger.info("Synced %d teams", len(teams))

    async def sync_schedule(self, game_date: date, session: Session) -> list[int]:
        """경기 일정을 DB에 UPSERT, game_pk 목록 반환"""
        raw_games = await self.fetch_schedule(game_date)

        # 선발 투수가 players 테이블에 없으면 stub 삽입 (FK 위반 방지)
        starter_ids: set[int] = set()
        for g in raw_games:
            for side in ("home", "away"):
                pid = (g.get("teams", {}).get(side, {}).get("probablePitcher") or {}).get("id")
                if pid:
                    starter_ids.add(int(pid))
        ensure_players_exist(starter_ids, session)

        game_pks = []
        for g in raw_games:
            teams = g.get("teams", {})
            home = teams.get("home", {})
            away = teams.get("away", {})
            venue = g.get("venue", {})
            game_dt_str = g.get("gameDate", "")
            game_dt = datetime.fromisoformat(game_dt_str.replace("Z", "+00:00")) if game_dt_str else None

            stmt = insert(Game).values(
                game_pk=g["gamePk"],
                game_date=game_date,
                game_datetime=game_dt,
                home_team_id=home.get("team", {}).get("id"),
                away_team_id=away.get("team", {}).get("id"),
                status=g.get("status", {}).get("abstractGameState", "scheduled"),
                venue_id=venue.get("id"),
                venue_name=(venue.get("name", "") or "")[:100],
                home_starter_id=(home.get("probablePitcher", {}) or {}).get("id"),
                away_starter_id=(away.get("probablePitcher", {}) or {}).get("id"),
                season=game_date.year,
            ).on_conflict_do_update(
                index_elements=["game_pk"],
                set_=dict(
                    status=g.get("status", {}).get("abstractGameState", "scheduled"),
                    home_starter_id=(home.get("probablePitcher", {}) or {}).get("id"),
                    away_starter_id=(away.get("probablePitcher", {}) or {}).get("id"),
                ),
            )
            session.execute(stmt)
            game_pks.append(g["gamePk"])
        session.commit()
        logger.info("Synced %d games for %s", len(game_pks), game_date)
        return game_pks

    async def update_game_results(self, game_date: date, session: Session) -> None:
        """전일 경기 결과(점수) 업데이트"""
        raw_games = await self.fetch_schedule(game_date)
        for g in raw_games:
            if g.get("status", {}).get("abstractGameState") != "Final":
                continue
            teams = g.get("teams", {})
            session.query(Game).filter(Game.game_pk == g["gamePk"]).update(
                {
                    "home_score": teams.get("home", {}).get("score"),
                    "away_score": teams.get("away", {}).get("score"),
                    "status": "Final",
                }
            )
        session.commit()
        logger.info("Updated results for %s", game_date)

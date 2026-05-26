"""경기 종료 직후 boxscore → pitcher/batter game logs UPSERT."""
from __future__ import annotations

from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from src.collector.base import BaseCollector
from src.common.logger import get_logger
from src.db.models.games import BatterGameLog, Game, PitcherGameLog
from src.db.models.predictions import GamePrediction
from src.db.player_utils import ensure_players_exist

logger = get_logger(__name__)


def _parse_ip(ip_str: object) -> float:
    """'5.2' → 5.667 (2/3 이닝). '0.0' → 0. None/잘못된 입력 → 0."""
    if ip_str is None:
        return 0.0
    s = str(ip_str)
    try:
        whole_str, _, frac_str = s.partition(".")
        whole = int(whole_str) if whole_str else 0
        frac = int(frac_str) if frac_str else 0
        return float(whole) + frac / 3.0
    except (TypeError, ValueError):
        return 0.0


def _extract_logs(box: dict, game: Game) -> tuple[list[dict], list[dict], int, int, set[int]]:
    """순수 함수: boxscore JSON → (투수로그 rows, 타자로그 rows, 홈점수, 원정점수, 선수ID set)."""
    teams = box.get("teams", {}) or {}
    pitch_rows: list[dict] = []
    bat_rows: list[dict] = []
    all_pids: set[int] = set()
    scores = {"home": 0, "away": 0}

    for side in ("home", "away"):
        tblock = teams.get(side) or {}
        team_id = ((tblock.get("team") or {}).get("id")
                   or (game.home_team_id if side == "home" else game.away_team_id))
        scores[side] = int(((tblock.get("teamStats") or {}).get("batting") or {}).get("runs", 0))

        players = tblock.get("players") or {}
        for _, p in players.items():
            pid_raw = (p.get("person") or {}).get("id")
            if not pid_raw:
                continue
            pid = int(pid_raw)
            all_pids.add(pid)
            stats = p.get("stats") or {}
            pstats = stats.get("pitching") or {}
            bstats = stats.get("batting") or {}

            ip = _parse_ip(pstats.get("inningsPitched"))
            if ip > 0:
                pitch_rows.append({
                    "player_id": pid, "game_pk": game.game_pk, "team_id": team_id,
                    "game_date": game.game_date, "season": game.season,
                    "is_starter": int(pstats.get("gamesStarted", 0) or 0) > 0,
                    "ip": ip,
                    "er": int(pstats.get("earnedRuns", 0) or 0),
                    "k": int(pstats.get("strikeOuts", 0) or 0),
                    "bb": int(pstats.get("baseOnBalls", 0) or 0),
                    "h": int(pstats.get("hits", 0) or 0),
                    "hr": int(pstats.get("homeRuns", 0) or 0),
                    "pitches": int(pstats.get("numberOfPitches", 0) or 0),
                })

            ab = int(bstats.get("atBats", 0) or 0)
            if ab > 0:
                bat_rows.append({
                    "player_id": pid, "game_pk": game.game_pk, "team_id": team_id,
                    "game_date": game.game_date, "season": game.season,
                    "ab": ab,
                    "h": int(bstats.get("hits", 0) or 0),
                    "hr": int(bstats.get("homeRuns", 0) or 0),
                    "rbi": int(bstats.get("rbi", 0) or 0),
                    "bb": int(bstats.get("baseOnBalls", 0) or 0),
                    "k": int(bstats.get("strikeOuts", 0) or 0),
                })

    return pitch_rows, bat_rows, scores["home"], scores["away"], all_pids


class PostgameCollector(BaseCollector):
    BASE = "https://statsapi.mlb.com/api/v1"

    async def fetch_boxscore(self, game_pk: int) -> dict:
        data = await self._get(f"{self.BASE}/game/{game_pk}/boxscore", timeout=15)
        assert isinstance(data, dict)
        return data

    async def sync_full(self, game_pk: int, session: Session) -> None:
        box = await self.fetch_boxscore(game_pk)
        game = session.query(Game).filter(Game.game_pk == game_pk).first()
        if not game:
            logger.warning("postgame: game %d not in DB", game_pk)
            return

        pitch_rows, bat_rows, h_score, a_score, all_pids = _extract_logs(box, game)

        # 1) games 최종 스코어 + Final
        session.query(Game).filter(Game.game_pk == game_pk).update({
            "home_score": h_score,
            "away_score": a_score,
            "status": "Final",
        })

        # 2) FK 위반 방지
        if all_pids:
            ensure_players_exist(all_pids, session)

        # 3) pitcher/batter game logs UPSERT (no-op on conflict)
        if pitch_rows:
            session.execute(
                insert(PitcherGameLog).values(pitch_rows).on_conflict_do_nothing(
                    constraint="uq_pitcher_game_logs_pid_pk"
                )
            )
        if bat_rows:
            session.execute(
                insert(BatterGameLog).values(bat_rows).on_conflict_do_nothing(
                    constraint="uq_batter_game_logs_pid_pk"
                )
            )

        # 4) 예측 채점
        if h_score != a_score:
            pred = session.query(GamePrediction).filter(
                GamePrediction.game_pk == game_pk
            ).first()
            if pred and pred.is_correct is None:
                home_won = h_score > a_score
                pick_home = pred.home_win_prob >= 0.5
                pred.is_correct = 1 if (home_won == pick_home) else 0

        session.commit()
        logger.info(
            "postgame game=%d final=%d-%d pitch_logs=%d bat_logs=%d",
            game_pk, h_score, a_score, len(pitch_rows), len(bat_rows),
        )

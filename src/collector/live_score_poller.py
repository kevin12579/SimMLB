"""1분 간격 라이브 스코어 폴러 + 라이브 win probability 재계산."""
from __future__ import annotations

import json
from datetime import datetime
from typing import TypedDict

from sqlalchemy import text
from sqlalchemy.orm import Session

from src.collector.base import BaseCollector
from src.common.logger import get_logger
from src.db.models.predictions import GamePrediction
from src.ml.live_win_prob import live_win_prob_adjuster

logger = get_logger(__name__)


class LiveState(TypedDict):
    game_pk: int
    status: str
    inning: int
    half: str
    outs: int
    balls: int
    strikes: int
    home_score: int
    away_score: int
    on1: bool
    on2: bool
    on3: bool
    mlb_wp: float | None


def _parse_state(game_pk: int, linescore: dict) -> LiveState:
    """순수 함수: linescore JSON → LiveState dict. 단위 테스트에서 mock 사용."""
    ls = linescore or {}
    teams = ls.get("teams") or {}
    home_team = teams.get("home") or {}
    away_team = teams.get("away") or {}
    offense = ls.get("offense") or {}

    mlb_wp_raw = home_team.get("winProb")
    mlb_wp = float(mlb_wp_raw) / 100.0 if isinstance(mlb_wp_raw, (int, float)) else None

    return {
        "game_pk": game_pk,
        "status": str((ls.get("status") or {}).get("abstractGameState")
                      or ls.get("gameStatus") or "Preview"),
        "inning": int(ls.get("currentInning") or 0),
        "half": str(ls.get("inningHalf") or "").lower(),
        "outs": int(ls.get("outs") or 0),
        "balls": int(ls.get("balls") or 0),
        "strikes": int(ls.get("strikes") or 0),
        "home_score": int(home_team.get("runs") or 0),
        "away_score": int(away_team.get("runs") or 0),
        "on1": bool(offense.get("first")),
        "on2": bool(offense.get("second")),
        "on3": bool(offense.get("third")),
        "mlb_wp": mlb_wp,
    }


class LiveScorePoller(BaseCollector):
    """단일 게임 polling 1회. Final 감지 시 True 반환."""

    BASE = "https://statsapi.mlb.com/api/v1"

    async def fetch_linescore(self, game_pk: int) -> dict:
        # /linescore 엔드포인트는 status를 자체 필드로 안 주므로 schedule로 한번 더 확인
        data = await self._get(f"{self.BASE}/game/{game_pk}/linescore", timeout=8)
        assert isinstance(data, dict)
        if "status" not in data:
            # status가 비어있으면 game endpoint로 폴백 (가벼운 호출)
            try:
                game_data = await self._get(
                    f"{self.BASE}/game/{game_pk}/content/summary", timeout=8
                )
                if isinstance(game_data, dict):
                    data.setdefault("status", {})
                    data["status"]["abstractGameState"] = (
                        game_data.get("editorial", {})
                                 .get("recap", {})
                                 .get("home", {})
                                 .get("seoTitle")
                    )
            except Exception:
                pass
        return data

    async def fetch_status(self, game_pk: int) -> str:
        """schedule 엔드포인트로 abstractGameState 정확히 조회."""
        data = await self._get(
            f"{self.BASE}/schedule", params={"sportId": 1, "gamePk": game_pk}, timeout=8
        )
        if not isinstance(data, dict):
            return "Preview"
        for d in data.get("dates", []):
            for g in d.get("games", []):
                if g.get("gamePk") == game_pk:
                    return str((g.get("status") or {}).get("abstractGameState", "Preview"))
        return "Preview"

    async def poll_once(self, game_pk: int, session: Session) -> bool:
        """한 번 polling → DB INSERT/UPDATE. status='Final'이면 True."""
        ls = await self.fetch_linescore(game_pk)
        # status가 linescore에 없으면 schedule에서 받아옴
        state = _parse_state(game_pk, ls)
        if state["status"] in ("", "Preview"):
            state["status"] = await self.fetch_status(game_pk)

        # 베이스 (T-15 pre-game 예측)
        base_row = session.execute(
            text("SELECT home_win_prob FROM game_predictions WHERE game_pk = :pk"),
            {"pk": game_pk},
        ).fetchone()
        base_prob = float(base_row[0]) if base_row else 0.5

        live_prob = live_win_prob_adjuster(
            base_prob=base_prob,
            inning=state["inning"], half=state["half"], outs=state["outs"],
            home_score=state["home_score"], away_score=state["away_score"],
            on1=state["on1"], on2=state["on2"], on3=state["on3"],
            mlb_wp=state["mlb_wp"],
        )

        # 1) game_live_states INSERT
        session.execute(text("""
            INSERT INTO game_live_states
              (game_pk, polled_at, game_status, current_inning, inning_half,
               outs, balls, strikes, home_score, away_score,
               on_first, on_second, on_third, mlb_win_prob, live_home_prob)
            VALUES (:pk, NOW(), :st, :inn, :half, :outs, :b, :s, :hs, :as_,
                    :o1, :o2, :o3, :mlb, :live)
        """), {
            "pk": game_pk, "st": state["status"],
            "inn": state["inning"], "half": state["half"],
            "outs": state["outs"], "b": state["balls"], "s": state["strikes"],
            "hs": state["home_score"], "as_": state["away_score"],
            "o1": state["on1"], "o2": state["on2"], "o3": state["on3"],
            "mlb": state["mlb_wp"], "live": live_prob,
        })

        # 2) game_predictions 최신 스냅샷 갱신
        session.query(GamePrediction).filter(GamePrediction.game_pk == game_pk).update({
            "live_home_win_prob": live_prob,
            "live_status": state["status"],
            "live_current_inning": state["inning"],
            "live_score_home": state["home_score"],
            "live_score_away": state["away_score"],
            "live_updated_at": datetime.utcnow(),
        })
        session.commit()

        # 3) Redis pub/sub (frontend 푸시)
        _publish_live(game_pk, state, live_prob)

        is_final = state["status"] == "Final"
        logger.info(
            "live poll game=%d status=%s inn=%d %d-%d live_prob=%.3f%s",
            game_pk, state["status"], state["inning"],
            state["home_score"], state["away_score"], live_prob,
            " (FINAL)" if is_final else "",
        )
        return is_final


def _publish_live(game_pk: int, state: LiveState, live_prob: float) -> None:
    try:
        import redis
        from config.settings import settings
        r = redis.from_url(f"redis://{settings.redis_host}:{settings.redis_port}")
        r.publish(f"live:{game_pk}", json.dumps({
            "game_pk": game_pk,
            "status": state["status"],
            "inning": state["inning"], "half": state["half"],
            "home_score": state["home_score"], "away_score": state["away_score"],
            "live_home_prob": live_prob,
        }))
    except Exception as e:
        logger.debug("Redis publish skipped: %s", e)

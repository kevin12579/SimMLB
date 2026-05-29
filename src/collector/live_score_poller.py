"""30초 간격 라이브 폴러 — MLB Live Feed 전환 + 타석 이벤트 감지 + Redis SSE push."""
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

# 마지막 play_id 추적 (per game_pk) — 새 타석 결과만 publish
_last_play_ids: dict[int, str] = {}


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
    play_id: str
    play_event: str
    play_desc: str


def _parse_state(game_pk: int, feed: dict) -> LiveState:
    """Live Feed JSON → LiveState dict."""
    gd = feed.get("gameData", {})
    ld = feed.get("liveData", {})
    ls = ld.get("linescore", {})
    teams_ls = ls.get("teams", {})
    offense = ls.get("offense", {})

    home_ls = teams_ls.get("home", {})
    away_ls = teams_ls.get("away", {})

    # MLB 공식 winProb (linescore 내 home 팀)
    mlb_wp_raw = home_ls.get("winProbability") or home_ls.get("winProb")
    mlb_wp: float | None = None
    if isinstance(mlb_wp_raw, (int, float)):
        v = float(mlb_wp_raw)
        mlb_wp = v / 100.0 if v > 1.0 else v  # 0~100 혹은 0~1 모두 대응

    # 현재 타석 이벤트
    plays = ld.get("plays", {})
    current_play = plays.get("currentPlay") or {}
    play_id: str = current_play.get("playId") or current_play.get("atBatIndex", "") or ""
    play_result = current_play.get("result") or {}
    play_event: str = play_result.get("event") or ""
    play_desc: str = play_result.get("description") or ""

    # 게임 상태
    status_obj = gd.get("status", {})
    abstract_state = status_obj.get("abstractGameState", "Preview")

    return {
        "game_pk": game_pk,
        "status": abstract_state,
        "inning": int(ls.get("currentInning") or 0),
        "half": str(ls.get("inningHalf") or ls.get("inningState") or "").lower(),
        "outs": int(ls.get("outs") or 0),
        "balls": int(ls.get("balls") or 0),
        "strikes": int(ls.get("strikes") or 0),
        "home_score": int(home_ls.get("runs") or 0),
        "away_score": int(away_ls.get("runs") or 0),
        "on1": bool(offense.get("first")),
        "on2": bool(offense.get("second")),
        "on3": bool(offense.get("third")),
        "mlb_wp": mlb_wp,
        "play_id": str(play_id),
        "play_event": play_event,
        "play_desc": play_desc,
    }


class LiveScorePoller(BaseCollector):
    """단일 게임 polling 1회. Final 감지 시 True 반환."""

    FEED_URL = "https://statsapi.mlb.com/api/v1.1/game/{pk}/feed/live"

    async def fetch_live_feed(self, game_pk: int) -> dict:
        data = await self._get(self.FEED_URL.format(pk=game_pk), timeout=10)
        assert isinstance(data, dict)
        return data

    async def poll_once(self, game_pk: int, session: Session) -> bool:
        """한 번 polling → DB INSERT/UPDATE. status='Final'이면 True."""
        feed = await self.fetch_live_feed(game_pk)
        state = _parse_state(game_pk, feed)

        # 베이스 (T-15 pre-game 예측)
        base_row = session.execute(
            text("SELECT home_win_prob FROM game_predictions WHERE game_pk = :pk"),
            {"pk": game_pk},
        ).fetchone()
        base_prob = float(base_row[0]) if base_row else 0.5

        live_prob = live_win_prob_adjuster(
            base_prob=base_prob,
            inning=state["inning"],
            half=state["half"],
            outs=state["outs"],
            home_score=state["home_score"],
            away_score=state["away_score"],
            on1=state["on1"],
            on2=state["on2"],
            on3=state["on3"],
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

        # 3) 새 타석 결과가 있을 때만 publish (or 30초마다 heartbeat publish)
        prev_play_id = _last_play_ids.get(game_pk, "")
        is_new_play = state["play_id"] and state["play_id"] != prev_play_id
        if is_new_play:
            _last_play_ids[game_pk] = state["play_id"]
        _publish_live(game_pk, state, live_prob, base_prob, is_new_play)

        is_final = state["status"] == "Final"
        logger.info(
            "live poll game=%d status=%s inn=%d %d-%d live_prob=%.3f mlb_wp=%s event=%s%s",
            game_pk, state["status"], state["inning"],
            state["home_score"], state["away_score"],
            live_prob, state["mlb_wp"], state["play_event"] or "-",
            " (FINAL)" if is_final else "",
        )
        return is_final


def _publish_live(
    game_pk: int,
    state: LiveState,
    live_prob: float,
    base_prob: float,
    is_new_play: bool,
) -> None:
    try:
        import redis
        from config.settings import settings

        r = redis.from_url(f"redis://{settings.redis_host}:{settings.redis_port}")
        payload = {
            "game_pk": game_pk,
            "status": state["status"],
            "inning": state["inning"],
            "half": state["half"],
            "outs": state["outs"],
            "balls": state["balls"],
            "strikes": state["strikes"],
            "home_score": state["home_score"],
            "away_score": state["away_score"],
            "on1": state["on1"],
            "on2": state["on2"],
            "on3": state["on3"],
            "mlb_wp": state["mlb_wp"],
            "live_home_prob": live_prob,
            "base_prob": base_prob,
            "play_event": state["play_event"],
            "play_desc": state["play_desc"],
            "is_new_play": is_new_play,
        }
        r.publish(f"live:{game_pk}", json.dumps(payload))
        # REST 엔드포인트가 읽을 수 있도록 최신 live_home_prob 저장 (120초 TTL)
        r.setex(f"live_prob:{game_pk}", 120, str(live_prob))
    except Exception as e:
        logger.debug("Redis publish skipped: %s", e)

import json
from datetime import date, datetime
from typing import Any

import redis.asyncio as aioredis
from fastapi import APIRouter, HTTPException
from sqlalchemy import text

from config.settings import settings
from src.db.models.games import Game, GameLineup
from src.db.models.players import Player
from src.db.models.predictions import GamePrediction
from src.db.models.teams import Team
from src.db.session import get_session

router = APIRouter()

_redis: aioredis.Redis | None = None


def _get_redis() -> aioredis.Redis:
    global _redis
    if _redis is None:
        _redis = aioredis.from_url(
            f"redis://{settings.redis_host}:{settings.redis_port}",
            decode_responses=True,
        )
    return _redis


def _serialize(obj: object) -> object:
    if isinstance(obj, (date, datetime)):
        return obj.isoformat()
    return obj


def _lineup_summary(session, game_pk: int, is_home: bool, limit: int = 3) -> list[str]:
    """라인업 상위 N명 이름 반환."""
    rows = session.execute(text("""
        SELECT p.full_name
        FROM game_lineups gl
        LEFT JOIN players p ON p.mlbam_id = gl.player_id
        WHERE gl.game_pk = :pk AND gl.is_home = :home
        ORDER BY gl.batting_order
        LIMIT :lim
    """), {"pk": game_pk, "home": is_home, "lim": limit}).fetchall()
    return [r[0] or "TBD" for r in rows]


def _build_game_payload(pred: GamePrediction, game: Game, team_map: dict, session) -> dict:
    return {
        "game_pk": pred.game_pk,
        "home_team": team_map.get(game.home_team_id, ""),
        "away_team": team_map.get(game.away_team_id, ""),
        "home_win_prob": round(pred.home_win_prob, 3),
        "away_win_prob": round(pred.away_win_prob, 3),
        "confidence": pred.confidence_level,
        "reasoning": pred.reasoning_text or "",
        "top5_features": pred.shap_top5 or [],
        "model_version": pred.model_version,
        # v2: weather
        "weather_temp_f": pred.weather_temp_f,
        "weather_condition": pred.weather_condition,
        "weather_wind": pred.weather_wind,
        # v2: lineup
        "home_lineup_preview": _lineup_summary(session, pred.game_pk, is_home=True),
        "away_lineup_preview": _lineup_summary(session, pred.game_pk, is_home=False),
        # v2: live snapshot
        "live": {
            "status": pred.live_status,
            "home_win_prob": (
                round(pred.live_home_win_prob, 3)
                if pred.live_home_win_prob is not None else None
            ),
            "current_inning": pred.live_current_inning,
            "score_home": pred.live_score_home,
            "score_away": pred.live_score_away,
            "updated_at": pred.live_updated_at.isoformat() if pred.live_updated_at else None,
        },
    }


@router.get("/today")
async def get_today_predictions() -> dict:
    today = date.today()
    cache_key = f"predictions:today:{today}"
    redis = _get_redis()

    cached = await redis.get(cache_key)
    if cached:
        return json.loads(cached)

    with get_session() as session:
        preds = session.query(GamePrediction).filter(
            GamePrediction.prediction_date == today
        ).all()
        team_map = {t.mlbam_team_id: t.abbreviation for t in session.query(Team).all()}
        game_map = {
            g.game_pk: g for g in session.query(Game).filter(Game.game_date == today).all()
        }

        games_list = []
        for p in preds:
            g = game_map.get(p.game_pk)
            if not g:
                continue
            games_list.append(_build_game_payload(p, g, team_map, session))

    result = {"date": today.isoformat(), "count": len(games_list), "games": games_list}
    await redis.setex(cache_key, 3600, json.dumps(result, default=_serialize))
    return result


@router.get("/{game_pk}")
async def get_game_prediction(game_pk: int) -> dict:
    cache_key = f"prediction:{game_pk}"
    redis = _get_redis()
    cached = await redis.get(cache_key)
    if cached:
        return json.loads(cached)

    with get_session() as session:
        pred = session.query(GamePrediction).filter(
            GamePrediction.game_pk == game_pk
        ).first()
        if not pred:
            raise HTTPException(status_code=404, detail=f"game_pk={game_pk} 예측 없음")
        game = session.query(Game).filter(Game.game_pk == game_pk).first()
        if not game:
            raise HTTPException(status_code=404, detail=f"game_pk={game_pk} game 없음")
        team_map = {t.mlbam_team_id: t.abbreviation for t in session.query(Team).all()}

        result: dict[str, Any] = _build_game_payload(pred, game, team_map, session)
        result.update({
            "lgbm_prob": round(pred.lgbm_prob or 0, 3),
            "xgb_prob": round(pred.xgb_prob or 0, 3),
            "prediction_date": pred.prediction_date.isoformat() if pred.prediction_date else "",
        })

    # 라이브 변동 가능 — 라이브 데이터 있는 경기는 짧은 TTL
    ttl = 60 if pred.live_status == "In Progress" else 3600
    await redis.setex(cache_key, ttl, json.dumps(result, default=_serialize))
    return result


@router.get("/{game_pk}/live")
async def get_game_live_states(game_pk: int, limit: int = 200) -> dict:
    """game_live_states 시계열 — frontend 라이브 차트용. 캐시 없음 (자주 변경)."""
    with get_session() as session:
        rows = session.execute(text("""
            SELECT polled_at, game_status, current_inning, inning_half,
                   outs, balls, strikes, home_score, away_score,
                   on_first, on_second, on_third, mlb_win_prob, live_home_prob
            FROM game_live_states
            WHERE game_pk = :pk
            ORDER BY polled_at ASC
            LIMIT :lim
        """), {"pk": game_pk, "lim": limit}).fetchall()

    return {
        "game_pk": game_pk,
        "count": len(rows),
        "states": [
            {
                "polled_at": r.polled_at.isoformat() if r.polled_at else None,
                "status": r.game_status,
                "inning": r.current_inning,
                "half": r.inning_half,
                "outs": r.outs,
                "balls": r.balls,
                "strikes": r.strikes,
                "home_score": r.home_score,
                "away_score": r.away_score,
                "bases": {"first": r.on_first, "second": r.on_second, "third": r.on_third},
                "mlb_win_prob": r.mlb_win_prob,
                "live_home_prob": r.live_home_prob,
            }
            for r in rows
        ],
    }

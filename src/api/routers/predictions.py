import json
from datetime import date, datetime, timezone

import redis.asyncio as aioredis
from fastapi import APIRouter, HTTPException
from sqlalchemy import select

from src.db.session import get_session
from src.db.models.predictions import GamePrediction
from src.db.models.games import Game
from src.db.models.teams import Team
from config.settings import settings

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


@router.get("/today")
async def get_today_predictions() -> dict:
    cache_key = f"predictions:today:{date.today()}"
    redis = _get_redis()

    cached = await redis.get(cache_key)
    if cached:
        return json.loads(cached)

    today = date.today()
    with get_session() as session:
        preds = session.query(GamePrediction).filter(
            GamePrediction.prediction_date == today
        ).all()
        team_map = {t.mlbam_team_id: t.abbreviation for t in session.query(Team).all()}
        game_map = {g.game_pk: g for g in session.query(Game).filter(
            Game.game_date == today
        ).all()}

        games_list = []
        for p in preds:
            g = game_map.get(p.game_pk)
            if not g:
                continue
            games_list.append({
                "game_pk":       p.game_pk,
                "home_team":     team_map.get(g.home_team_id, ""),
                "away_team":     team_map.get(g.away_team_id, ""),
                "home_win_prob": round(p.home_win_prob, 3),
                "away_win_prob": round(p.away_win_prob, 3),
                "confidence":    p.confidence_level,
                "reasoning":     p.reasoning_text or "",
                "top5_features": p.shap_top5 or [],
            })

    result = {"date": today.isoformat(), "games": games_list}
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

        game     = session.query(Game).filter(Game.game_pk == game_pk).first()
        team_map = {t.mlbam_team_id: t.abbreviation for t in session.query(Team).all()}

        result = {
            "game_pk":       pred.game_pk,
            "home_team":     team_map.get(game.home_team_id, "") if game else "",
            "away_team":     team_map.get(game.away_team_id, "") if game else "",
            "home_win_prob": round(pred.home_win_prob, 3),
            "away_win_prob": round(pred.away_win_prob, 3),
            "confidence":    pred.confidence_level,
            "reasoning":     pred.reasoning_text or "",
            "top5_features": pred.shap_top5 or [],
            "model_version": pred.model_version,
            "lgbm_prob":     round(pred.lgbm_prob or 0, 3),
            "xgb_prob":      round(pred.xgb_prob  or 0, 3),
            "prediction_date": pred.prediction_date.isoformat() if pred.prediction_date else "",
        }

    await redis.setex(cache_key, 3600, json.dumps(result))
    return result

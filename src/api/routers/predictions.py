import json
from datetime import date, datetime, timedelta
from typing import Any

try:
    from zoneinfo import ZoneInfo
except ImportError:
    import pytz as ZoneInfo  # type: ignore

import redis.asyncio as aioredis
from fastapi import APIRouter, HTTPException
from sqlalchemy import text

_KST = ZoneInfo("Asia/Seoul")

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
        "game_datetime": game.game_datetime.isoformat() if game.game_datetime else None,
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
    today = datetime.now(_KST).date()
    cache_key = f"predictions:today:{today}"
    redis = _get_redis()

    cached = await redis.get(cache_key)
    if cached:
        return json.loads(cached)

    # KST 오늘(today) 경기 = MLB US 날짜 today-1로 저장됨
    us_today = today - timedelta(days=1)

    with get_session() as session:
        team_map = {t.mlbam_team_id: t.abbreviation for t in session.query(Team).all()}
        games_today = session.query(Game).filter(Game.game_date == us_today).all()
        game_map = {g.game_pk: g for g in games_today}
        us_game_pks = list(game_map.keys())

        preds = (
            session.query(GamePrediction)
            .filter(GamePrediction.game_pk.in_(us_game_pks))
            .all()
        ) if us_game_pks else []

        games_list = []
        for p in preds:
            g = game_map.get(p.game_pk)
            if not g:
                continue
            games_list.append(_build_game_payload(p, g, team_map, session))

    result = {"date": today.isoformat(), "count": len(games_list), "games": games_list}
    await redis.setex(cache_key, 3600, json.dumps(result, default=_serialize))
    return result


@router.get("/history")
async def get_predictions_history(days: int = 7) -> dict:
    """최근 N일 예측 이력 — ScreenHistory / ScreenModel 용."""
    kst_today = datetime.now(_KST).date()
    us_end = kst_today - timedelta(days=1)
    us_start = kst_today - timedelta(days=days)

    with get_session() as session:
        team_map = {t.mlbam_team_id: t.abbreviation for t in session.query(Team).all()}
        games = {
            g.game_pk: g
            for g in session.query(Game).filter(
                Game.game_date >= us_start,
                Game.game_date <= us_end,
            ).all()
        }
        if not games:
            return {"days": days, "total": 0, "graded": 0, "correct": 0,
                    "accuracy": None, "brier": None, "rows": []}

        preds = session.query(GamePrediction).filter(
            GamePrediction.game_pk.in_(list(games.keys()))
        ).all()

        rows = []
        for p in preds:
            g = games.get(p.game_pk)
            if not g:
                continue
            home = team_map.get(g.home_team_id, "")
            away = team_map.get(g.away_team_id, "")
            pick_home = p.home_win_prob >= 0.5
            pick_team = home if pick_home else away
            pick_prob = p.home_win_prob if pick_home else p.away_win_prob
            rows.append({
                "game_pk": p.game_pk,
                "date": (g.game_date + timedelta(days=1)).isoformat(),
                "game_datetime": g.game_datetime.isoformat() if g.game_datetime else None,
                "home_team": home,
                "away_team": away,
                "home_score": g.home_score,
                "away_score": g.away_score,
                "status": g.status,
                "home_win_prob": round(p.home_win_prob, 3),
                "away_win_prob": round(p.away_win_prob, 3),
                "confidence": p.confidence_level,
                "is_correct": p.is_correct,
                "pick_team": pick_team,
                "pick_prob": round(pick_prob, 3),
            })

    graded = [r for r in rows if r["is_correct"] is not None]
    correct = sum(1 for r in graded if r["is_correct"] == 1)
    brier = None
    if graded:
        brier = round(
            sum((r["pick_prob"] - r["is_correct"]) ** 2 for r in graded) / len(graded), 4
        )

    return {
        "days": days,
        "total": len(rows),
        "graded": len(graded),
        "correct": correct,
        "accuracy": round(correct / len(graded) * 100, 1) if graded else None,
        "brier": brier,
        "rows": rows,
    }


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

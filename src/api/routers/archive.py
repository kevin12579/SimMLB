import json
from datetime import date, datetime
from fastapi import APIRouter
from sqlalchemy import text
import redis.asyncio as aioredis
from config.settings import settings
from src.db.session import get_session
from src.db.models.games import Game
from src.db.models.players import Player
from src.db.models.predictions import GamePrediction
from src.db.models.teams import Team

router = APIRouter()

_redis: aioredis.Redis | None = None
def _get_redis():
    global _redis
    if _redis is None:
        _redis = aioredis.from_url(f"redis://{settings.redis_host}:{settings.redis_port}", decode_responses=True)
    return _redis

def _ser(obj):
    if isinstance(obj, (date, datetime)): return obj.isoformat()
    return obj

@router.get("/summary")
async def archive_summary(target_date: str):
    try:
        d = date.fromisoformat(target_date)
    except ValueError:
        return {"error": "날짜 형식이 올바르지 않습니다 (YYYY-MM-DD)"}

    cache_key = f"archive:{target_date}"
    redis = _get_redis()
    cached = await redis.get(cache_key)
    if cached:
        return json.loads(cached)

    with get_session() as session:
        team_map = {t.mlbam_team_id: t.abbreviation for t in session.query(Team).all()}

        preds = session.query(GamePrediction).filter(
            GamePrediction.prediction_date == d
        ).all()
        game_pks = [p.game_pk for p in preds]
        games = {g.game_pk: g for g in session.query(Game).filter(Game.game_pk.in_(game_pks)).all()}

        # 선발 투수 이름 조회
        starter_ids = {
            sid for g in games.values()
            for sid in (g.home_starter_id, g.away_starter_id) if sid
        }
        starter_map: dict[int, str] = {}
        if starter_ids:
            for p in session.query(Player).filter(Player.mlbam_id.in_(starter_ids)).all():
                starter_map[p.mlbam_id] = p.full_name

        games_list = []
        graded = correct = high_med_total = high_med_correct = 0

        for p in preds:
            g = games.get(p.game_pk)
            if not g: continue
            home = team_map.get(g.home_team_id, "")
            away = team_map.get(g.away_team_id, "")
            pick_home = p.home_win_prob >= 0.5
            pick_team = home if pick_home else away
            has_result = g.home_score is not None and g.away_score is not None
            actual_winner = None
            if has_result:
                actual_winner = home if g.home_score > g.away_score else away

            is_correct = p.is_correct
            if is_correct is not None:
                graded += 1
                if is_correct == 1: correct += 1
                if p.confidence_level in ['HIGH','MED']:
                    high_med_total += 1
                    if is_correct == 1: high_med_correct += 1

            games_list.append({
                "game_pk": p.game_pk,
                "home_team": home, "away_team": away,
                "home_score": g.home_score, "away_score": g.away_score,
                "status": g.status,
                "game_datetime": g.game_datetime.isoformat() if g.game_datetime else None,
                "venue_name": g.venue_name,
                "home_win_prob": round(p.home_win_prob, 3),
                "away_win_prob": round(p.away_win_prob, 3),
                "confidence": p.confidence_level,
                "pick_team": pick_team,
                "actual_winner": actual_winner,
                "is_correct": is_correct,
                "model_version": p.model_version,
                "home_starter_name": starter_map.get(g.home_starter_id or 0, ""),
                "away_starter_name": starter_map.get(g.away_starter_id or 0, ""),
            })

        result = {
            "date": target_date,
            "total": len(games_list),
            "graded": graded,
            "correct": correct,
            "accuracy": round(correct/graded*100,1) if graded else None,
            "high_med_accuracy": round(high_med_correct/high_med_total*100,1) if high_med_total else None,
            "games": games_list,
        }

    ttl = 300 if d < date.today() else 60
    await redis.setex(cache_key, ttl, json.dumps(result, default=_ser))
    return result

@router.get("/calendar")
async def archive_calendar(year: int, month: int):
    """월별 달력 데이터 — 날짜별 경기수/적중률"""
    cache_key = f"archive:cal:{year}:{month}"
    redis = _get_redis()
    cached = await redis.get(cache_key)
    if cached: return json.loads(cached)

    with get_session() as session:
        rows = session.execute(text(f"""
            SELECT prediction_date,
                   COUNT(*) as total,
                   SUM(CASE WHEN is_correct=1 THEN 1 ELSE 0 END) as correct,
                   SUM(CASE WHEN is_correct IS NOT NULL THEN 1 ELSE 0 END) as graded
            FROM game_predictions
            WHERE EXTRACT(YEAR FROM prediction_date)={year}
              AND EXTRACT(MONTH FROM prediction_date)={month}
            GROUP BY prediction_date
            ORDER BY prediction_date
        """)).fetchall()

    cal = [{"date": str(r[0]), "total": r[1], "correct": r[2], "graded": r[3],
            "accuracy": round(r[2]/r[3]*100,1) if r[3] else None} for r in rows]
    result = {"year": year, "month": month, "days": cal}
    await redis.setex(cache_key, 600, json.dumps(result, default=_ser))
    return result
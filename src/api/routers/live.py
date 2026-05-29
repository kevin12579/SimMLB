import asyncio
import json
from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
import aiohttp
import redis.asyncio as aioredis
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


@router.get("/game/{game_pk}")
async def get_live_game(game_pk: int):
    cache_key = f"live_game:{game_pk}"
    redis = _get_redis()
    cached = await redis.get(cache_key)
    if cached:
        return json.loads(cached)

    url = f"https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=8)) as resp:
                if resp.status != 200:
                    return {"status": "OFFLINE", "game_pk": game_pk}
                data = await resp.json()
    except Exception:
        return {"status": "OFFLINE", "game_pk": game_pk}

    gd = data.get("gameData", {})
    ld = data.get("liveData", {})
    ls = ld.get("linescore", {})
    teams = gd.get("teams", {})

    result = {
        "game_pk": game_pk,
        "status": gd.get("status", {}).get("abstractGameState", "Preview"),
        "detailed_state": gd.get("status", {}).get("detailedState", ""),
        "current_inning": ls.get("currentInning"),
        "inning_state": ls.get("inningState", ""),
        "balls": ls.get("balls", 0),
        "strikes": ls.get("strikes", 0),
        "outs": ls.get("outs", 0),
        "runs": {
            "home": ls.get("teams", {}).get("home", {}).get("runs", 0),
            "away": ls.get("teams", {}).get("away", {}).get("runs", 0),
        },
        "hits": {
            "home": ls.get("teams", {}).get("home", {}).get("hits", 0),
            "away": ls.get("teams", {}).get("away", {}).get("hits", 0),
        },
        "errors": {
            "home": ls.get("teams", {}).get("home", {}).get("errors", 0),
            "away": ls.get("teams", {}).get("away", {}).get("errors", 0),
        },
        "runners": {
            "first":  "first"  in ls.get("offense", {}),
            "second": "second" in ls.get("offense", {}),
            "third":  "third"  in ls.get("offense", {}),
        },
        "home_team": teams.get("home", {}).get("abbreviation", ""),
        "away_team": teams.get("away", {}).get("abbreviation", ""),
        "home_name": teams.get("home", {}).get("teamName", ""),
        "away_name": teams.get("away", {}).get("teamName", ""),
        "venue": gd.get("venue", {}).get("name", ""),
        "pitchers": {
            "home_probable": ((gd.get("probablePitchers") or {}).get("home") or {}).get("fullName", ""),
            "away_probable": ((gd.get("probablePitchers") or {}).get("away") or {}).get("fullName", ""),
            "current": (ls.get("defense", {}).get("pitcher") or {}).get("fullName", ""),
            "winner": (ld.get("decisions", {}).get("winner") or {}).get("fullName", ""),
            "loser": (ld.get("decisions", {}).get("loser") or {}).get("fullName", ""),
        },
    }

    # 폴러가 저장한 최신 live_home_prob 조회 (없으면 생략)
    live_prob_raw = await redis.get(f"live_prob:{game_pk}")
    if live_prob_raw:
        result["live_home_prob"] = float(live_prob_raw)

    await redis.setex(cache_key, 10, json.dumps(result))
    return result


@router.get("/stream/{game_pk}")
async def stream_live(game_pk: int, request: Request):
    """SSE 엔드포인트 — Redis pub/sub live:{game_pk} 채널을 구독해 프론트에 push."""
    async def event_generator():
        r = _get_redis()
        pubsub = r.pubsub()
        await pubsub.subscribe(f"live:{game_pk}")
        try:
            yield f"data: {json.dumps({'type': 'connected', 'game_pk': game_pk})}\n\n"
            last_heartbeat = asyncio.get_event_loop().time()
            while True:
                if await request.is_disconnected():
                    break
                try:
                    msg = await asyncio.wait_for(
                        pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0),
                        timeout=2.0,
                    )
                except asyncio.TimeoutError:
                    msg = None

                if msg and msg.get("type") == "message":
                    yield f"data: {msg['data']}\n\n"
                    last_heartbeat = asyncio.get_event_loop().time()

                # 30초마다 keepalive
                now = asyncio.get_event_loop().time()
                if now - last_heartbeat > 30:
                    yield ": heartbeat\n\n"
                    last_heartbeat = now

        except asyncio.CancelledError:
            pass
        finally:
            await pubsub.unsubscribe(f"live:{game_pk}")
            await pubsub.aclose()

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )

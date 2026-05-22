import json
from fastapi import APIRouter
import aiohttp
import redis.asyncio as aioredis
from config.settings import settings

router = APIRouter()

_redis: aioredis.Redis | None = None
def _get_redis():
    global _redis
    if _redis is None:
        _redis = aioredis.from_url(
            f"redis://{settings.redis_host}:{settings.redis_port}",
            decode_responses=True
        )
    return _redis

LEAGUE_IDS = {"AL": 103, "NL": 104}

def _split_rec(split_records, type_):
    for s in split_records:
        if s.get("type") == type_:
            return s.get("wins", 0), s.get("losses", 0)
    return 0, 0

@router.get("")
async def get_standings(season: int = 2026):
    cache_key = f"standings3:{season}"
    redis = _get_redis()
    cached = await redis.get(cache_key)
    if cached:
        return json.loads(cached)

    result = {
        "AL": {"East": [], "Central": [], "West": []},
        "NL": {"East": [], "Central": [], "West": []},
        "wildcard": {"AL": [], "NL": []}
    }

    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        async with aiohttp.ClientSession(headers=headers) as sess:
            for lg, lid in LEAGUE_IDS.items():
                url = (
                    f"https://statsapi.mlb.com/api/v1/standings"
                    f"?leagueId={lid}&season={season}&standingsTypes=regularSeason"
                    f"&hydrate=division,conference,sport,league,team"
                )
                try:
                    async with sess.get(url, timeout=aiohttp.ClientTimeout(total=12)) as resp:
                        if resp.status != 200:
                            continue
                        data = await resp.json(content_type=None)
                except Exception as e:
                    continue

                all_teams = []
                div_leaders: set = set()

                for div_rec in data.get("records", []):
                    div_name = div_rec.get("division", {}).get("name", "")
                    div_abbr = div_rec.get("division", {}).get("nameShort", div_name)

                    if "East" in div_name:
                        div_key = "East"
                    elif "Central" in div_name:
                        div_key = "Central"
                    elif "West" in div_name:
                        div_key = "West"
                    else:
                        div_key = "East"

                    team_recs = div_rec.get("teamRecords", [])
                    for i, tr in enumerate(team_recs):
                        team = tr.get("team", {})
                        splits = tr.get("records", {}).get("splitRecords", [])
                        hw, hl = _split_rec(splits, "home")
                        aw, al = _split_rec(splits, "away")
                        l10w, l10l = _split_rec(splits, "lastTen")
                        streak = tr.get("streak", {}).get("streakCode", "—")
                        div_rank = tr.get("divisionRank", str(i + 1))
                        wc_rank = tr.get("wildCardRank", "—")
                        wc_gb = tr.get("wildCardGamesBack", "—")
                        gb = tr.get("gamesBack", "—")

                        entry = {
                            "team_id":   team.get("id", 0),
                            "team_name": team.get("teamName", team.get("name", "")),
                            "city":      team.get("locationName", team.get("franchiseName", "")),
                            "abbr":      team.get("abbreviation", ""),
                            "division":  div_key,
                            "league":    lg,
                            "wins":      tr.get("wins", 0),
                            "losses":    tr.get("losses", 0),
                            "win_pct":   float(tr.get("winningPercentage", "0") or 0),
                            "gb":        gb,
                            "streak":    streak,
                            "home_w": hw, "home_l": hl,
                            "away_w": aw, "away_l": al,
                            "l10_w": l10w, "l10_l": l10l,
                            "div_rank":  int(div_rank) if str(div_rank).isdigit() else i + 1,
                            "wc_rank":   int(wc_rank) if str(wc_rank).isdigit() else 99,
                            "wc_gb":     wc_gb,
                        }
                        result[lg][div_key].append(entry)
                        all_teams.append(entry)
                        if i == 0:
                            div_leaders.add(team.get("id", 0))

                # 각 디비전 정렬
                for div_key in result[lg]:
                    result[lg][div_key].sort(key=lambda x: x["div_rank"])

                # 와일드카드: 디비전 1위 제외, wc_rank 순
                wc_teams = [t for t in all_teams if t["team_id"] not in div_leaders]
                wc_teams.sort(key=lambda x: x["wc_rank"])
                result["wildcard"][lg] = wc_teams

    except Exception as e:
        return {"error": str(e), **result}

    await redis.setex(cache_key, 1800, json.dumps(result))
    return result
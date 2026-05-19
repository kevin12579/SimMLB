"""컨텍스트 피처 (4개): 구장, 날씨"""
import json
from pathlib import Path

from sqlalchemy.orm import Session
from sqlalchemy import text


def _load_park_factors() -> dict:
    path = Path("config/park_factors.json")
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


PARK_FACTORS = _load_park_factors()


def get_context_features(game_pk: int, venue_id: int | None, session: Session) -> dict:
    pf = PARK_FACTORS.get(str(venue_id), {"run_factor": 1.0})

    # weather
    weather_row = session.execute(text("""
        SELECT temp_f, wind_speed_mph, g.is_dome
        FROM game_weather w
        JOIN games g ON g.game_pk = w.game_pk
        WHERE w.game_pk = :gpk
    """), {"gpk": game_pk}).fetchone()

    if weather_row:
        temp_f         = float(weather_row.temp_f or 72.0)
        wind_speed_mph = float(weather_row.wind_speed_mph or 0.0)
        is_dome        = bool(weather_row.is_dome)
    else:
        temp_f         = 72.0
        wind_speed_mph = 0.0
        is_dome        = False

    if is_dome:
        temp_f         = 72.0
        wind_speed_mph = 0.0

    return {
        "park_run_factor":  float(pf.get("run_factor", 1.0)),
        "is_dome":          float(is_dome),
        "temp_f":           temp_f,
        "wind_speed_mph":   wind_speed_mph,
    }

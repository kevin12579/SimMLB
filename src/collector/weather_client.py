"""Open-Meteo API로 경기장 날씨 수집 (무료, 인증 불필요)"""
from datetime import date

from sqlalchemy.orm import Session
from sqlalchemy.dialects.postgresql import insert

from src.collector.base import BaseCollector
from src.db.models.games import Game, GameWeather
from src.common.logger import get_logger

logger = get_logger(__name__)

# 구장별 위경도 (주요 구장)
VENUE_COORDS: dict[int, tuple[float, float]] = {
    31: (34.0739, -118.2400),   # Dodger Stadium
    3313: (40.8296, -73.9262),  # Yankee Stadium
    2392: (42.3467, -71.0972),  # Fenway Park
    2681: (41.8299, -87.6338),  # Wrigley Field
    2394: (37.7786, -122.3893), # Oracle Park
    2395: (47.5914, -122.3326), # T-Mobile Park
    5325: (39.7559, -104.9942), # Coors Field
    2680: (33.4453, -112.0667), # Chase Field (dome)
    15: (25.7781, -80.2197),    # Marlins Park (dome)
    680: (27.7682, -82.6534),   # Tropicana Field (dome)
}

DOME_VENUES = {2680, 15, 680, 4169, 2889}  # 돔 구장 venue_id

BASE_URL = "https://api.open-meteo.com/v1/forecast"


class WeatherClient(BaseCollector):
    async def fetch_game_weather(self, game_date: date, venue_id: int) -> dict | None:
        coords = VENUE_COORDS.get(venue_id)
        if not coords:
            logger.debug("No coords for venue_id %d", venue_id)
            return None

        lat, lon = coords
        data = await self._get(
            BASE_URL,
            params={
                "latitude": lat,
                "longitude": lon,
                "hourly": "temperature_2m,wind_speed_10m,precipitation",
                "temperature_unit": "fahrenheit",
                "wind_speed_unit": "mph",
                "start_date": game_date.isoformat(),
                "end_date": game_date.isoformat(),
                "timezone": "America/New_York",
            },
        )
        hourly = data.get("hourly", {})
        # 오후 7시 (19시) 기준 날씨 사용
        idx = 19
        temps = hourly.get("temperature_2m", [])
        winds = hourly.get("wind_speed_10m", [])
        precips = hourly.get("precipitation", [])
        return {
            "temp_f": temps[idx] if len(temps) > idx else None,
            "wind_speed_mph": winds[idx] if len(winds) > idx else None,
            "precip_mm": precips[idx] if len(precips) > idx else None,
        }

    async def sync_weather_for_date(self, game_date: date, session: Session) -> None:
        """당일 경기 날씨를 모두 수집해서 DB 저장"""
        games = session.query(Game).filter(Game.game_date == game_date).all()
        for game in games:
            is_dome = (game.venue_id or 0) in DOME_VENUES
            if is_dome:
                weather = {"temp_f": 72.0, "wind_speed_mph": 0.0, "precip_mm": 0.0}
            else:
                weather = await self.fetch_game_weather(game_date, game.venue_id or 0)

            if weather is None:
                continue

            stmt = insert(GameWeather).values(
                game_pk=game.game_pk,
                temp_f=weather.get("temp_f"),
                wind_speed_mph=weather.get("wind_speed_mph"),
                precip_mm=weather.get("precip_mm"),
            ).on_conflict_do_update(
                index_elements=["game_pk"],
                set_=weather,
            )
            session.execute(stmt)

            # Game 테이블의 is_dome 업데이트
            session.query(Game).filter(Game.game_pk == game.game_pk).update(
                {"is_dome": is_dome}
            )

        session.commit()
        logger.info("Synced weather for %d games on %s", len(games), game_date)

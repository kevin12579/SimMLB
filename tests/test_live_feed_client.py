"""LiveFeedClient.parse_snapshot 순수 함수 단위 테스트."""
from src.collector.live_feed_client import LiveFeedClient, _parse_temp


class TestParseTemp:
    def test_numeric_string(self):
        assert _parse_temp("72") == 72.0
        assert _parse_temp("68.5") == 68.5

    def test_none_or_blank(self):
        assert _parse_temp(None) is None
        assert _parse_temp("") is None
        assert _parse_temp("--") is None
        assert _parse_temp("N/A") is None

    def test_invalid(self):
        assert _parse_temp("hot") is None


class TestParseSnapshot:
    def test_full_payload(self):
        c = LiveFeedClient()
        live = {
            "gameData": {
                "weather": {"temp": "72", "condition": "Clear", "wind": "5 mph, In From RF"},
                "probablePitchers": {
                    "home": {"id": 100001, "fullName": "Home Starter"},
                    "away": {"id": 100002, "fullName": "Away Starter"},
                },
            },
            "liveData": {
                "boxscore": {
                    "teams": {
                        "home": {"battingOrder": [200001, 200002, 200003,
                                                  200004, 200005, 200006,
                                                  200007, 200008, 200009]},
                        "away": {"battingOrder": [300001, 300002, 300003,
                                                  300004, 300005, 300006,
                                                  300007, 300008, 300009]},
                    },
                },
            },
        }
        snap = c.parse_snapshot(999, live)
        assert snap["game_pk"] == 999
        assert snap["weather_temp_f"] == 72.0
        assert snap["weather_condition"] == "Clear"
        assert "5 mph" in snap["weather_wind"]
        assert snap["home_starter_id"] == 100001
        assert snap["away_starter_id"] == 100002
        assert len(snap["home_lineup_ids"]) == 9
        assert len(snap["away_lineup_ids"]) == 9
        assert snap["home_lineup_ids"][0] == 200001

    def test_empty_payload(self):
        c = LiveFeedClient()
        snap = c.parse_snapshot(1, {})
        assert snap["weather_temp_f"] is None
        assert snap["weather_condition"] == ""
        assert snap["home_starter_id"] is None
        assert snap["home_lineup_ids"] == []

    def test_lineup_not_announced(self):
        c = LiveFeedClient()
        live = {
            "gameData": {
                "weather": {"temp": "--", "condition": "Roof Closed", "wind": ""},
                "probablePitchers": {"home": {"id": 555}},  # away starter unknown
            },
            "liveData": {"boxscore": {"teams": {"home": {}, "away": {}}}},
        }
        snap = c.parse_snapshot(2, live)
        assert snap["weather_temp_f"] is None
        assert snap["weather_condition"] == "Roof Closed"
        assert snap["home_starter_id"] == 555
        assert snap["away_starter_id"] is None
        assert snap["home_lineup_ids"] == []

    def test_malformed_lineup_ids_coerced(self):
        c = LiveFeedClient()
        live = {
            "liveData": {
                "boxscore": {
                    "teams": {
                        "home": {"battingOrder": ["111", "222"]},  # strings
                        "away": {"battingOrder": []},
                    }
                }
            }
        }
        snap = c.parse_snapshot(3, live)
        assert snap["home_lineup_ids"] == [111, 222]

    def test_weather_strings_truncated(self):
        c = LiveFeedClient()
        long_cond = "x" * 200
        live = {"gameData": {"weather": {"temp": "70", "condition": long_cond, "wind": "x" * 100}}}
        snap = c.parse_snapshot(4, live)
        assert len(snap["weather_condition"]) == 50
        assert len(snap["weather_wind"]) == 50

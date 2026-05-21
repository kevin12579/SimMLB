"""postgame_collector 순수 함수 단위 테스트."""
from datetime import date
from types import SimpleNamespace

from src.collector.postgame_collector import _extract_logs, _parse_ip


class TestParseIP:
    def test_whole_innings(self):
        assert _parse_ip("5.0") == 5.0

    def test_one_third(self):
        assert _parse_ip("5.1") == 5.0 + 1 / 3.0

    def test_two_thirds(self):
        assert _parse_ip("5.2") == 5.0 + 2 / 3.0

    def test_none_and_invalid(self):
        assert _parse_ip(None) == 0.0
        assert _parse_ip("hi") == 0.0
        assert _parse_ip("") == 0.0


def _make_game():
    return SimpleNamespace(
        game_pk=999,
        game_date=date(2025, 5, 21),
        season=2025,
        home_team_id=119,
        away_team_id=147,
    )


def _box(home_runs=5, away_runs=3, *, starter_pid=10001, batter_pid=20001):
    return {
        "teams": {
            "home": {
                "team": {"id": 119},
                "teamStats": {"batting": {"runs": home_runs}},
                "players": {
                    f"ID{starter_pid}": {
                        "person": {"id": starter_pid},
                        "stats": {
                            "pitching": {
                                "inningsPitched": "6.2",
                                "earnedRuns": 2, "strikeOuts": 7,
                                "baseOnBalls": 1, "hits": 5,
                                "homeRuns": 1, "numberOfPitches": 95,
                                "gamesStarted": 1,
                            },
                        },
                    },
                    f"ID{batter_pid}": {
                        "person": {"id": batter_pid},
                        "stats": {
                            "batting": {
                                "atBats": 4, "hits": 2, "homeRuns": 1,
                                "rbi": 3, "baseOnBalls": 1, "strikeOuts": 1,
                            },
                        },
                    },
                },
            },
            "away": {
                "team": {"id": 147},
                "teamStats": {"batting": {"runs": away_runs}},
                "players": {},
            },
        },
    }


class TestExtractLogs:
    def test_basic_extract(self):
        pitch, bat, h_score, a_score, pids = _extract_logs(_box(), _make_game())
        assert h_score == 5
        assert a_score == 3
        assert len(pitch) == 1
        assert len(bat) == 1
        assert 10001 in pids and 20001 in pids
        p = pitch[0]
        assert p["player_id"] == 10001
        assert p["ip"] == 6.0 + 2 / 3.0
        assert p["k"] == 7
        assert p["is_starter"] is True

    def test_pitcher_with_zero_ip_skipped(self):
        box = _box()
        box["teams"]["home"]["players"]["ID10001"]["stats"]["pitching"]["inningsPitched"] = "0.0"
        pitch, _, _, _, _ = _extract_logs(box, _make_game())
        assert len(pitch) == 0

    def test_batter_with_zero_ab_skipped(self):
        box = _box()
        box["teams"]["home"]["players"]["ID20001"]["stats"]["batting"]["atBats"] = 0
        _, bat, _, _, _ = _extract_logs(box, _make_game())
        assert len(bat) == 0

    def test_empty_teams(self):
        pitch, bat, h, a, pids = _extract_logs({"teams": {}}, _make_game())
        assert pitch == [] and bat == [] and pids == set()

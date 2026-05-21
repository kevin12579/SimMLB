"""LiveScorePoller._parse_state 순수 함수 단위 테스트."""
from src.collector.live_score_poller import _parse_state


class TestParseState:
    def test_full_payload(self):
        ls = {
            "status": {"abstractGameState": "Live"},
            "currentInning": 5,
            "inningHalf": "Bottom",
            "outs": 2,
            "balls": 3,
            "strikes": 1,
            "teams": {
                "home": {"runs": 4, "winProb": 65.0},
                "away": {"runs": 2},
            },
            "offense": {"first": True, "second": False, "third": True},
        }
        state = _parse_state(999, ls)
        assert state["status"] == "Live"
        assert state["inning"] == 5
        assert state["half"] == "bottom"
        assert state["home_score"] == 4
        assert state["away_score"] == 2
        assert state["on1"] is True
        assert state["on2"] is False
        assert state["on3"] is True
        assert state["mlb_wp"] == 0.65

    def test_pregame_minimal(self):
        state = _parse_state(1, {})
        assert state["inning"] == 0
        assert state["home_score"] == 0
        assert state["away_score"] == 0
        assert state["mlb_wp"] is None
        assert state["status"] == "Preview"

    def test_final(self):
        ls = {
            "status": {"abstractGameState": "Final"},
            "currentInning": 9,
            "teams": {"home": {"runs": 5}, "away": {"runs": 3}},
        }
        state = _parse_state(2, ls)
        assert state["status"] == "Final"
        assert state["home_score"] == 5

    def test_no_winprob(self):
        ls = {"teams": {"home": {"runs": 1}, "away": {"runs": 2}}}
        state = _parse_state(3, ls)
        assert state["mlb_wp"] is None

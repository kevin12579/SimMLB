"""update_dwar 순수 함수 단위 테스트."""
import pandas as pd

from scripts.update_dwar import _process


SAMPLE = pd.DataFrame({
    "player_ID": ["a1", "a1", "b1", "c1", "d1"],
    "year_ID":   [2024, 2025, 2025, 2024, 2025],
    "WAR_def":   [1.0, 2.0, 0.5, -0.3, 1.5],
    "WAR":       [3.0, 4.0, 2.0, 1.0, 5.0],
})


class TestProcess:
    def test_filter_by_season(self):
        out = _process(SAMPLE, [2025])
        assert set(out["season"]) == {2025}
        assert len(out) == 3  # a1, b1, d1

    def test_groupby_multi_team(self):
        # 멀티팀 선수 가정 — 같은 player_id, 같은 season 두 row
        df = pd.DataFrame({
            "player_ID": ["x1", "x1"],
            "year_ID":   [2025, 2025],
            "WAR_def":   [0.5, 0.7],
            "WAR":       [1.0, 2.0],
        })
        out = _process(df, [2025])
        assert len(out) == 1
        assert out.iloc[0]["dWAR"] == 1.2
        assert out.iloc[0]["total_WAR"] == 3.0

    def test_empty_returns_schema(self):
        out = _process(pd.DataFrame(), [2025])
        assert list(out.columns) == ["player_id", "season", "dWAR", "total_WAR"]
        assert len(out) == 0

    def test_no_war_def_column_defaults_zero(self):
        df = pd.DataFrame({
            "player_ID": ["a1"],
            "year_ID": [2025],
            "WAR": [3.0],
        })
        out = _process(df, [2025])
        assert out.iloc[0]["dWAR"] == 0.0

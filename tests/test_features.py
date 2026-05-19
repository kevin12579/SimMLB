"""Week 2 완료 기준 테스트: test_no_lookahead, test_shrinkage_edge_cases"""
import pytest
from datetime import date
from unittest.mock import MagicMock

from src.ml.features.team_features import shrunk_rate, pythagenpat_wp, get_team_features
from src.ml.features.pitcher_features import get_pitcher_features


# ──────────────────────────────────────────
# test_shrinkage_edge_cases
# ──────────────────────────────────────────

class TestShrinkageEdgeCases:
    def test_zero_games_returns_prior(self):
        assert shrunk_rate(0, 0) == 0.5

    def test_large_sample_converges_to_observed(self):
        result = shrunk_rate(80, 100)
        # 100경기 80승 → 실제 승률 0.80에 수렴: (80+10)/(100+20)=0.75 이상
        assert result >= 0.75

    def test_small_sample_shrinks_toward_prior(self):
        # 3경기 3승이어도 소표본이므로 0.80보다 훨씬 낮아야 함
        result = shrunk_rate(3, 3)
        assert result < 0.70

    def test_custom_prior(self):
        result = shrunk_rate(0, 0, prior=0.55)
        assert result == 0.55

    def test_pythagenpat_zero_games(self):
        assert pythagenpat_wp(0, 0, 0) == 0.5

    def test_pythagenpat_equal_runs(self):
        result = pythagenpat_wp(500, 500, 100)
        assert abs(result - 0.5) < 0.01

    def test_pythagenpat_dominant_offense(self):
        result = pythagenpat_wp(800, 400, 100)
        assert result > 0.70


# ──────────────────────────────────────────
# test_no_lookahead — 미래 데이터 누수 방지 검증
# ──────────────────────────────────────────

class TestNoLookahead:
    """as_of_date 이전 데이터만 사용하는지 검증 (SQL 시점 제한)"""

    def _make_session(self, return_value):
        session = MagicMock()
        row = MagicMock()
        row.wins = return_value.get("wins", 0)
        row.games = return_value.get("games", 0)
        row.rs = return_value.get("rs", 0)
        row.ra = return_value.get("ra", 0)
        row.hw = return_value.get("hw", 0)
        row.hg = return_value.get("hg", 0)
        row.w10 = return_value.get("w10", 0)
        row.rs10 = return_value.get("rs10", 0)
        row.ra10 = return_value.get("ra10", 0)
        row.last_date = return_value.get("last_date")
        session.execute.return_value.fetchone.return_value = row
        session.execute.return_value.fetchall.return_value = []
        return session

    def test_sql_contains_as_of_constraint(self):
        """get_team_features의 SQL이 game_date < :as_of 조건을 반드시 포함하는지 검증"""
        import inspect
        from src.ml.features import team_features
        source = inspect.getsource(team_features)
        assert "game_date < :as_of" in source, "시점 제한 조건 누락!"

    def test_pitcher_sql_contains_as_of_constraint(self):
        import inspect
        from src.ml.features import pitcher_features
        source = inspect.getsource(pitcher_features)
        assert "game_date < :as_of" in source, "투수 피처에 시점 제한 조건 누락!"

    def test_feature_keys_have_correct_prefix(self):
        """홈팀 피처는 'home_', 원정팀은 'away_' prefix를 가져야 함"""
        session = self._make_session({})
        home_feat = get_team_features(1, date(2024, 6, 1), is_home=True, session=session)
        away_feat = get_team_features(1, date(2024, 6, 1), is_home=False, session=session)
        assert all(k.startswith("home_") for k in home_feat)
        assert all(k.startswith("away_") for k in away_feat)

    def test_missing_pitcher_returns_defaults(self):
        """선발 투수 없음(None) → 기본값 반환, 에러 없음"""
        session = MagicMock()
        result = get_pitcher_features(None, date(2024, 6, 1), is_home=True, session=session)
        assert result["home_sp_season_era"] == 4.50
        assert not session.execute.called, "pitcher_id=None 시 DB 쿼리 하면 안 됨"

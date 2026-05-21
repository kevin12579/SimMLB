"""bref_features_v2 단위 테스트."""
import pandas as pd
import pytest

from src.ml.features import bref_features_v2 as v2


class TestFeatureColumns:
    def test_count_is_47(self):
        assert len(v2.FEATURE_COLS_V2) == 47

    def test_no_duplicates(self):
        assert len(set(v2.FEATURE_COLS_V2)) == 47

    def test_required_groups(self):
        cols = set(v2.FEATURE_COLS_V2)
        # 핵심 그룹별 존재
        assert "home_roll_win" in cols and "away_roll_win" in cols
        assert "home_starter_velo" in cols and "home_starter_whiff" in cols
        assert "home_bat_ev" in cols and "home_bat_hardhit" in cols
        assert "home_dwar" in cols and "away_dwar" in cols
        assert "home_rest" in cols and "away_rest" in cols
        assert "park_run_factor" in cols and "is_dome" in cols


def _sample_park():
    return {"LAD": {"run_factor": 1.05, "hr_factor": 1.10, "is_dome": False}}


def _sample_pitch():
    base = {"era": 3.50, "fip": 3.80, "whip": 1.20, "k9": 9.0, "bb9": 2.5}
    return {
        ("LAD", 2025): base,
        ("NYY", 2025): {**base, "era": 4.10},
    }


def _sample_bat():
    return {
        ("LAD", 2025): {"ops": 0.760, "obp": 0.330, "slg": 0.430,
                       "ba": 0.255, "hr_pa": 0.035, "dwar": 5.0},
        ("NYY", 2025): {"ops": 0.770, "obp": 0.335, "slg": 0.435,
                       "ba": 0.260, "hr_pa": 0.036, "dwar": 3.0},
    }


class TestBuildFeatureRowV2:
    def test_returns_47_keys(self):
        row = v2.build_feature_row_v2(
            h_abbr="LAD", a_abbr="NYY", season=2025,
            h_roll=0.60, a_roll=0.55,
            h_starter_id=100, a_starter_id=200,
            h_lineup_ids=[1, 2, 3], a_lineup_ids=[10, 20],
            h_team_id=119, a_team_id=147,
            pitch_team=_sample_pitch(),
            bat_team=_sample_bat(),
            park=_sample_park(),
            sc_pitcher_indiv={100: {"velo": 95.0, "spin": 2300.0, "whiff": 0.30},
                              200: {"velo": 92.0, "spin": 2100.0, "whiff": 0.22}},
            sc_team_bat={("LAD", 2025): {"ev": 89.0, "la": 13.0, "hard_hit": 0.40}},
            sc_batter_indiv={1: {"ev": 91.0, "la": 14.0, "hard_hit": 0.45},
                             2: {"ev": 89.0, "la": 12.0, "hard_hit": 0.40},
                             3: {"ev": 93.0, "la": 15.0, "hard_hit": 0.50}},
            rest_cache={119: 2, 147: 4},
        )
        assert set(row.keys()) == set(v2.FEATURE_COLS_V2)

    def test_roll_diff_correct(self):
        row = v2.build_feature_row_v2(
            h_abbr="LAD", a_abbr="NYY", season=2025,
            h_roll=0.65, a_roll=0.50,
            h_starter_id=None, a_starter_id=None,
            h_lineup_ids=[], a_lineup_ids=[],
            h_team_id=119, a_team_id=147,
            pitch_team=_sample_pitch(), bat_team=_sample_bat(), park=_sample_park(),
            sc_pitcher_indiv={}, sc_team_bat={}, sc_batter_indiv={},
            rest_cache={},
        )
        assert row["roll_win_diff"] == pytest.approx(0.15)
        assert row["era_diff"] == pytest.approx(4.10 - 3.50)
        assert row["ops_diff"] == pytest.approx(0.760 - 0.770)

    def test_lineup_avg_uses_individual_when_matched(self):
        row = v2.build_feature_row_v2(
            h_abbr="LAD", a_abbr="NYY", season=2025,
            h_roll=0.5, a_roll=0.5,
            h_starter_id=None, a_starter_id=None,
            h_lineup_ids=[1, 2], a_lineup_ids=[],
            h_team_id=119, a_team_id=147,
            pitch_team=_sample_pitch(), bat_team=_sample_bat(), park=_sample_park(),
            sc_pitcher_indiv={},
            sc_team_bat={("LAD", 2025): {"ev": 80.0, "la": 5.0, "hard_hit": 0.10},
                         ("NYY", 2025): {"ev": 85.0, "la": 10.0, "hard_hit": 0.30}},
            sc_batter_indiv={1: {"ev": 92.0, "la": 14.0, "hard_hit": 0.45},
                             2: {"ev": 90.0, "la": 12.0, "hard_hit": 0.35}},
            rest_cache={},
        )
        # home은 라인업 매칭 → 개인 평균
        assert row["home_bat_ev"] == pytest.approx(91.0)
        assert row["home_bat_hardhit"] == pytest.approx(0.40)
        # away는 라인업 매칭 0 → 팀 평균 폴백
        assert row["away_bat_ev"] == pytest.approx(85.0)
        assert row["away_bat_hardhit"] == pytest.approx(0.30)

    def test_falls_back_to_league_when_starter_unknown(self):
        row = v2.build_feature_row_v2(
            h_abbr="LAD", a_abbr="NYY", season=2025,
            h_roll=0.5, a_roll=0.5,
            h_starter_id=None, a_starter_id=999999,  # 999999 가 사전에 없음
            h_lineup_ids=[], a_lineup_ids=[],
            h_team_id=119, a_team_id=147,
            pitch_team=_sample_pitch(), bat_team=_sample_bat(), park=_sample_park(),
            sc_pitcher_indiv={123: {"velo": 99, "spin": 3000, "whiff": 0.5}},
            sc_team_bat={}, sc_batter_indiv={},
            rest_cache={},
        )
        assert row["home_starter_velo"] == v2.LEAGUE_SC_PITCH["velo"]
        assert row["away_starter_velo"] == v2.LEAGUE_SC_PITCH["velo"]

    def test_season_fallback_when_target_missing(self):
        # 2026 데이터 없으면 2025로 폴백 (target_season = season -1)
        row = v2.build_feature_row_v2(
            h_abbr="LAD", a_abbr="NYY", season=2026,  # 2026 없음
            h_roll=0.5, a_roll=0.5,
            h_starter_id=None, a_starter_id=None,
            h_lineup_ids=[], a_lineup_ids=[],
            h_team_id=119, a_team_id=147,
            pitch_team=_sample_pitch(), bat_team=_sample_bat(), park=_sample_park(),
            sc_pitcher_indiv={}, sc_team_bat={}, sc_batter_indiv={},
            rest_cache={},
        )
        # 2025 LAD era 3.50 이 적용됐는지
        assert row["home_era"] == pytest.approx(3.50)
        assert row["home_dwar"] == pytest.approx(5.0)

    def test_park_factor_and_dome(self):
        row = v2.build_feature_row_v2(
            h_abbr="LAD", a_abbr="NYY", season=2025,
            h_roll=0.5, a_roll=0.5,
            h_starter_id=None, a_starter_id=None,
            h_lineup_ids=[], a_lineup_ids=[],
            h_team_id=119, a_team_id=147,
            pitch_team=_sample_pitch(), bat_team=_sample_bat(),
            park={"LAD": {"run_factor": 1.08, "hr_factor": 1.12, "is_dome": True}},
            sc_pitcher_indiv={}, sc_team_bat={}, sc_batter_indiv={},
            rest_cache={119: 3, 147: 1},
        )
        assert row["park_run_factor"] == pytest.approx(1.08)
        assert row["park_hr_factor"] == pytest.approx(1.12)
        assert row["is_dome"] == 1.0
        assert row["home_rest"] == 3.0
        assert row["away_rest"] == 1.0


class TestLoadBrefBattingV2:
    def test_loads_dwar(self, tmp_path, monkeypatch):
        csv = tmp_path / "bref_batting_2025.csv"
        csv.write_text(
            "Rk,Team,PA,AB,H,HR,BB,HBP,SF,TB,dWAR\n"
            "1,LAD,500,450,130,25,40,3,5,200,4.5\n"
            "2,NYY,490,440,125,30,35,4,4,210,2.0\n"
        )
        monkeypatch.setattr(v2, "DATA_RAW", tmp_path)
        result = v2.load_bref_batting_v2(seasons=[2025])
        assert ("LAD", 2025) in result
        assert result[("LAD", 2025)]["dwar"] == pytest.approx(4.5)
        assert result[("NYY", 2025)]["dwar"] == pytest.approx(2.0)
        # OPS = OBP + SLG  계산 검증
        lad = result[("LAD", 2025)]
        expected_obp = (130 + 40 + 3) / (450 + 40 + 3 + 5)
        expected_slg = 200 / 450
        assert lad["obp"] == pytest.approx(expected_obp)
        assert lad["slg"] == pytest.approx(expected_slg)
        assert lad["ops"] == pytest.approx(expected_obp + expected_slg)

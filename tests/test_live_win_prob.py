"""live_win_prob 단위 테스트."""
import pytest

from src.ml.live_win_prob import live_win_prob_adjuster, _lookup_lead


class TestLookupLead:
    def test_exact_match(self):
        assert _lookup_lead(5, 3) == 0.88

    def test_closest_inning_below(self):
        # (4, +3) 키는 없음 → (3, +3) 사용
        assert _lookup_lead(4, 3) == 0.83

    def test_negative_diff_inverted(self):
        # (5, -3) → 1 - 0.88 = 0.12
        assert _lookup_lead(5, -3) == pytest.approx(0.12)

    def test_zero_diff_returns_none(self):
        assert _lookup_lead(5, 0) is None

    def test_no_match_returns_none(self):
        assert _lookup_lead(1, 7) is None  # diff=7 키 없음


class TestLiveWinProbAdjuster:
    def _kw(self, **overrides):
        defaults = dict(
            base_prob=0.55, inning=0, half="", outs=0,
            home_score=0, away_score=0,
            on1=False, on2=False, on3=False, mlb_wp=None,
        )
        defaults.update(overrides)
        return defaults

    def test_pregame_returns_base(self):
        p = live_win_prob_adjuster(**self._kw(inning=0))
        assert p == 0.55
        # 1회 0:0 0아웃 top도 pregame 취급
        p2 = live_win_prob_adjuster(**self._kw(inning=1, outs=0, half="top"))
        assert p2 == 0.55

    def test_mlb_wp_wins(self):
        p = live_win_prob_adjuster(**self._kw(
            inning=5, home_score=2, away_score=1, mlb_wp=0.72,
        ))
        assert p == 0.72

    def test_mlb_wp_out_of_range_ignored(self):
        # 0.0001은 클램프 안되고 룩업으로 폴백
        p = live_win_prob_adjuster(**self._kw(
            inning=5, home_score=2, away_score=1, mlb_wp=0.0001,
        ))
        # diff=+1 inning=5 → 룩업 0.69 + base 가중평균
        assert 0.55 < p < 0.75

    def test_lookup_blended_with_base(self):
        p = live_win_prob_adjuster(**self._kw(
            base_prob=0.50, inning=7, home_score=3, away_score=0,
        ))
        # diff=+3 inning=7 → 룩업 0.94. w_look = min(0.9, 0.3+0.07*7)=0.79
        # blended = 0.21*0.5 + 0.79*0.94 = 0.105 + 0.7426 = 0.8476
        assert 0.84 < p < 0.86

    def test_score_tied_falls_back_to_base_decay(self):
        p = live_win_prob_adjuster(**self._kw(
            base_prob=0.7, inning=5, home_score=2, away_score=2,
        ))
        # 룩업 없음 → progress=5/9=0.556 → decay 30%
        # blended = 0.7*(1-0.556*0.3) + 0.5*0.556*0.3
        expected = 0.7 * (1 - 0.556 * 0.3) + 0.5 * 0.556 * 0.3
        assert p == pytest.approx(expected, abs=0.005)

    def test_clamp_extremes(self):
        # mlb_wp 0.0001 무시 + lookup blended가 0.01~0.99 클램프 적용
        p = live_win_prob_adjuster(**self._kw(
            base_prob=0.01, inning=9, home_score=0, away_score=5,
        ))
        assert 0.01 <= p <= 0.99

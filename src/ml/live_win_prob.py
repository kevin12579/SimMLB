"""베이스 pre-game 예측 + 현재 경기 상황 → 라이브 win probability.

전략:
  1) MLB가 제공하는 winProbability 우선 (가능 시)
  2) 자체 룩업 폴백 (Tom Tango "The Book" 근사)
  3) 룩업 없으면 base 가중평균
"""
from __future__ import annotations

# 이닝/홈-원정 점수차 기반 간이 룩업: 홈팀 승률
_LEAD_TABLE: dict[tuple[int, int], float] = {
    (1, 1): 0.61, (1, 3): 0.78, (1, 5): 0.91,
    (3, 1): 0.64, (3, 3): 0.83, (3, 5): 0.94,
    (5, 1): 0.69, (5, 3): 0.88, (5, 5): 0.97,
    (7, 1): 0.78, (7, 3): 0.94, (7, 5): 0.99,
    (9, 1): 0.95, (9, 3): 0.998,
}


def _lookup_lead(inning: int, diff: int) -> float | None:
    """가장 가까운 (≤ inning, abs(diff)) 키 찾기. 음수 diff는 1 - lookup(+diff)."""
    if diff == 0:
        return None
    sign = 1 if diff > 0 else -1
    abs_diff = abs(diff)
    best: tuple[int, float] | None = None
    for (i, d), wp in _LEAD_TABLE.items():
        if d == abs_diff and i <= inning:
            if best is None or i > best[0]:
                best = (i, wp)
    if best is None:
        return None
    return best[1] if sign > 0 else 1.0 - best[1]


def live_win_prob_adjuster(
    *,
    base_prob: float,
    inning: int,
    half: str,
    outs: int,
    home_score: int,
    away_score: int,
    on1: bool,
    on2: bool,
    on3: bool,
    mlb_wp: float | None,
) -> float:
    """경기 진행도와 점수차이 반영한 홈팀 라이브 승률 (0.01~0.99 클램프)."""
    # 경기 시작 전 (1회 0:0 0아웃)
    if inning == 0 or (inning == 1 and home_score == 0 and away_score == 0 and outs == 0
                       and half in ("", "top")):
        return round(base_prob, 4)

    # 1) MLB 공식 winProbability 우선
    if mlb_wp is not None and 0.001 <= mlb_wp <= 0.999:
        return round(mlb_wp, 4)

    # 2) 룩업 테이블 폴백
    diff = home_score - away_score
    looked = _lookup_lead(inning, diff)

    if looked is None:
        # 점수차 0 or 룩업 미스 → base 감쇠 (이닝 진행도에 비례)
        progress = min(1.0, inning / 9.0)
        blended = base_prob * (1.0 - progress * 0.3) + 0.5 * progress * 0.3
        return round(max(0.01, min(0.99, blended)), 4)

    # 3) base와 lookup의 가중평균 — 이닝 진행도에 따라 lookup 가중치 증가
    w_look = min(0.9, 0.3 + 0.07 * inning)  # 1회 0.37 → 9회 0.93
    w_base = 1.0 - w_look
    blended = w_base * base_prob + w_look * looked
    return round(max(0.01, min(0.99, blended)), 4)

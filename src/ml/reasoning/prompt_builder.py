"""SHAP top-5 피처로 LLM 프롬프트 생성"""


FEATURE_LABELS: dict[str, str] = {
    "home_sp_fip":              "홈 선발 FIP",
    "away_sp_fip":              "원정 선발 FIP",
    "home_sp_xfip":             "홈 선발 xFIP",
    "away_sp_xfip":             "원정 선발 xFIP",
    "home_sp_last3_era":        "홈 선발 최근 3선발 ERA",
    "away_sp_last3_era":        "원정 선발 최근 3선발 ERA",
    "home_last10_win_rate":     "홈팀 최근 10경기 승률",
    "away_last10_win_rate":     "원정팀 최근 10경기 승률",
    "home_team_xwoba":          "홈팀 xwOBA",
    "away_team_xwoba":          "원정팀 xwOBA",
    "home_team_barrel_rate":    "홈팀 배럴 비율",
    "home_lineup_wrc_plus":     "홈팀 라인업 wRC+",
    "away_lineup_wrc_plus":     "원정팀 라인업 wRC+",
    "park_run_factor":          "구장 득점 팩터",
    "rest_diff":                "홈팀 휴식일 차이",
    "home_pythagenpat_wp":      "홈팀 Pythagorean 기대승률",
    "home_streak_signed":       "홈팀 연승/연패",
}


def build_reasoning_prompt(
    home_team: str,
    away_team: str,
    home_win_prob: float,
    shap_top5: list[dict],
) -> str:
    feature_lines = []
    for item in shap_top5:
        name  = item.get("feature", "")
        value = item.get("value", 0)
        shap  = item.get("shap_value", 0)
        label = FEATURE_LABELS.get(name, name)
        direction = "유리" if shap > 0 else "불리"
        feature_lines.append(f"- {label}: {value:.3f} ({direction})")

    features_text = "\n".join(feature_lines)
    fav = home_team if home_win_prob >= 0.5 else away_team
    fav_prob = home_win_prob if home_win_prob >= 0.5 else 1 - home_win_prob

    return f"""당신은 MLB 야구 분석 전문가입니다. 아래 데이터를 바탕으로 한국어로 2~3문장의 경기 분석을 작성하세요.

경기: {away_team} @ {home_team}
홈팀 승리 확률: {home_win_prob:.1%}
예상 승리팀: {fav} ({fav_prob:.1%})

주요 피처 (SHAP 기여도 상위 5개):
{features_text}

지침:
- 수치를 구체적으로 언급하세요
- 전문 야구 용어를 자연스럽게 사용하세요
- 2~3문장으로 간결하게 작성하세요
- 마크다운, 이모지 사용 금지"""


async def generate_reasoning(
    home_team: str,
    away_team: str,
    home_win_prob: float,
    shap_top5: list[dict],
) -> str:
    from src.ml.reasoning.llm_client import get_llm_client
    client = get_llm_client()
    prompt = build_reasoning_prompt(home_team, away_team, home_win_prob, shap_top5)
    try:
        return await client.complete(prompt)
    except Exception as e:
        from src.common.logger import get_logger
        get_logger(__name__).warning("LLM 근거 생성 실패: %s", e)
        return f"{home_team} 홈 승리 확률 {home_win_prob:.1%}로 예측됨."

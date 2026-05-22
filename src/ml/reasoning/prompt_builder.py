"""SHAP top-5 피처로 LLM 프롬프트 생성"""

FEATURE_LABELS: dict[str, str] = {
    "home_sp_fip":           "홈 선발 FIP",
    "away_sp_fip":           "원정 선발 FIP",
    "home_sp_era":           "홈 선발 ERA",
    "away_sp_era":           "원정 선발 ERA",
    "home_sp_xfip":          "홈 선발 xFIP",
    "away_sp_xfip":          "원정 선발 xFIP",
    "home_sp_last3_era":     "홈 선발 최근 3선발 ERA",
    "away_sp_last3_era":     "원정 선발 최근 3선발 ERA",
    "home_sp_whip":          "홈 선발 WHIP",
    "away_sp_whip":          "원정 선발 WHIP",
    "home_sp_k9":            "홈 선발 K/9",
    "away_sp_k9":            "원정 선발 K/9",
    "home_last10_win_rate":  "홈팀 최근 10경기 승률",
    "away_last10_win_rate":  "원정팀 최근 10경기 승률",
    "home_team_xwoba":       "홈팀 xwOBA",
    "away_team_xwoba":       "원정팀 xwOBA",
    "home_team_obp":         "홈팀 출루율(OBP)",
    "away_team_obp":         "원정팀 출루율(OBP)",
    "home_team_slg":         "홈팀 장타율(SLG)",
    "away_team_slg":         "원정팀 장타율(SLG)",
    "home_team_ba":          "홈팀 타율(BA)",
    "away_team_ba":          "원정팀 타율(BA)",
    "home_team_barrel_rate": "홈팀 배럴 비율",
    "away_team_barrel_rate": "원정팀 배럴 비율",
    "home_lineup_wrc_plus":  "홈팀 라인업 wRC+",
    "away_lineup_wrc_plus":  "원정팀 라인업 wRC+",
    "park_run_factor":       "구장 득점 팩터",
    "rest_diff":             "홈팀 휴식일 차이",
    "home_pythagenpat_wp":   "홈팀 Pythagorean 기대승률",
    "home_streak_signed":    "홈팀 연승/연패",
    "era_diff":              "ERA 차이(홈-원정)",
    "fip_diff":              "FIP 차이(홈-원정)",
    "obp_diff":              "OBP 차이(홈-원정)",
    "wrc_diff":              "wRC+ 차이(홈-원정)",
}

# 투수 관련 피처 키워드
PITCHER_FEATURES = {
    "fip", "era", "xfip", "whip", "k9", "last3_era"
}

# 타선 관련 피처 키워드
BATTER_FEATURES = {
    "xwoba", "obp", "slg", "ba", "barrel", "wrc"
}


def _categorize_features(shap_top5: list[dict]) -> dict:
    """피처를 투수/타선/기타로 분류"""
    pitcher = []
    batter = []
    other = []

    for item in shap_top5:
        name = item.get("feature", "").lower()
        label = FEATURE_LABELS.get(item.get("feature", ""), item.get("feature", ""))
        value = item.get("value", 0)
        shap = item.get("shap_value", 0)
        direction = "홈팀 유리" if shap > 0 else "원정팀 유리"

        entry = {
            "label": label,
            "value": value,
            "shap": shap,
            "direction": direction,
        }

        if any(k in name for k in PITCHER_FEATURES):
            pitcher.append(entry)
        elif any(k in name for k in BATTER_FEATURES):
            batter.append(entry)
        else:
            other.append(entry)

    return {"pitcher": pitcher, "batter": batter, "other": other}


def build_reasoning_prompt(
    home_team: str,
    away_team: str,
    home_win_prob: float,
    shap_top5: list[dict],
) -> str:
    fav = home_team if home_win_prob >= 0.5 else away_team
    und = away_team if home_win_prob >= 0.5 else home_team
    fav_prob = home_win_prob if home_win_prob >= 0.5 else 1 - home_win_prob
    away_prob = 1 - home_win_prob

    cats = _categorize_features(shap_top5)

    # 피처 상세 라인 구성
    all_lines = []
    for item in shap_top5:
        name = item.get("feature", "")
        value = item.get("value", 0)
        shap = item.get("shap_value", 0)
        label = FEATURE_LABELS.get(name, name)
        direction = "→ 홈팀 유리" if shap > 0 else "→ 원정팀 유리"
        all_lines.append(f"  · {label}: {value:.3f} (SHAP {shap:+.4f}, {direction})")

    features_detail = "\n".join(all_lines)

    # 투수 정보 요약
    pitcher_summary = ""
    if cats["pitcher"]:
        items = cats["pitcher"]
        pitcher_summary = "\n  ".join([
            f"{p['label']}: {p['value']:.3f} ({p['direction']})" for p in items
        ])
        pitcher_summary = f"\n[선발 투수 지표]\n  {pitcher_summary}"

    # 타선 정보 요약
    batter_summary = ""
    if cats["batter"]:
        items = cats["batter"]
        batter_summary = "\n  ".join([
            f"{b['label']}: {b['value']:.3f} ({b['direction']})" for b in items
        ])
        batter_summary = f"\n[타선 지표]\n  {batter_summary}"

    return f"""당신은 MLB 야구 전문 분석가입니다. 아래 통계 데이터를 바탕으로 오늘 경기를 분석하는 한국어 문단을 작성하세요.

=== 경기 정보 ===
매치업: {away_team}(원정) @ {home_team}(홈)
모델 예측: {fav} 승리 {fav_prob:.1%} / {und} 승리 {1-fav_prob:.1%}
홈팀 승리 확률: {home_win_prob:.1%} | 원정팀 승리 확률: {away_prob:.1%}

=== 주요 예측 근거 (SHAP 상위 5개) ==={pitcher_summary}{batter_summary}

[전체 피처 목록]
{features_detail}

=== 작성 지침 ===
1. 반드시 선발 투수 매치업을 언급하세요 (홈/원정 선발의 ERA, FIP, WHIP 등 수치 인용)
2. 유의미한 통계 수치(ERA, FIP, OBP, wRC+, 배럴률 등)를 2개 이상 구체적으로 인용하세요
3. SHAP 방향(홈팀 유리/원정팀 유리)을 바탕으로 어느 팀이 우세한지 명확하게 서술하세요
4. 마지막 문장에서 "{fav}의 승리가 유력하다" 또는 "{fav}에 우위가 있다"는 결론을 내리세요
5. 3~4문장으로 작성하세요
6. 마크다운, 이모지, 줄바꿈 기호 사용 금지
7. 팀명은 영문 약어({home_team}, {away_team})로 표기하세요"""


def build_advanced_analysis_prompt(
    home_team, away_team,
    h_starter, a_starter,
    h_era, a_era,
    h_fip, a_fip,
    h_whip, a_whip,
    h_k9, a_k9,
    h_ops, a_ops,
    h_obp, a_obp,
    h_slg, a_slg,
    h_roll, a_roll,
    h_velo, a_velo,
    h_hard_hit, a_hard_hit,
    h_bullpen_era, a_bullpen_era,
    home_win_prob, fav, fav_prob,
    features_text,
    park_name, park_run_factor, is_dome,
    h_last5_scores, a_last5_scores,
    h_injuries, a_injuries,
):
    away_win_prob = 1 - home_win_prob
    momentum_home = "상승세" if h_roll >= 0.6 else "하락세" if h_roll <= 0.4 else "중립"
    momentum_away = "상승세" if a_roll >= 0.6 else "하락세" if a_roll <= 0.4 else "중립"
    park_char = (
        "돔구장이라 날씨 변수는 없으며" if is_dome
        else f"파크팩터 {park_run_factor:.2f}의 {'타자 친화적' if park_run_factor > 1.05 else '투수 친화적' if park_run_factor < 0.95 else '중립적'} 구장으로"
    )

    return f"""당신은 메이저리그(MLB) 전문 퀀트 야구 분석가입니다.
아래에 제공된 ML 모델 예측값, SHAP 핵심 요인, 양 팀의 다층 스탯을 종합하여
베터와 야구팬이 읽었을 때 "이 경기 흐름을 완전히 이해했다"고 느낄 수 있는
깊이 있는 분석문을 작성하세요.

───────────────────────────────────────
[경기 기본 정보]
- 매치업: {away_team} (원정) @ {home_team} (홈)
- 구장: {park_name} ({park_char})
- 선발 투수: {a_starter} vs {h_starter}
- 모델 예측: {home_team} 승 {home_win_prob:.1%} / {away_team} 승 {away_win_prob:.1%}
- 최종 탑독: {fav} ({fav_prob:.1%})

───────────────────────────────────────
[선발 투수 심층 비교]
항목               {a_starter}(원정)   {h_starter}(홈)
ERA (시즌)         {a_era:.2f}              {h_era:.2f}
FIP (수비무관ERA)  {a_fip:.2f}              {h_fip:.2f}
WHIP               {a_whip:.2f}             {h_whip:.2f}
K/9 (탈삼진)       {a_k9:.1f}               {h_k9:.1f}
평균 구속           {a_velo:.1f}mph          {h_velo:.1f}mph
ERA-FIP 괴리       {abs(a_era-a_fip):.2f} ({'운 좋음' if a_era < a_fip else '운 나쁨'})      {abs(h_era-h_fip):.2f} ({'운 좋음' if h_era < h_fip else '운 나쁨'})

※ FIP > ERA : 현재 성적이 수비 도움을 많이 받은 것 → 앞으로 성적 악화 가능성
※ FIP < ERA : 현재 성적이 저평가 → 앞으로 성적 개선 가능성

───────────────────────────────────────
[타선 체급 비교]
항목          {away_team}(원정)   {home_team}(홈)
OPS           {a_ops:.3f}              {h_ops:.3f}
OBP           {a_obp:.3f}              {h_obp:.3f}
SLG           {a_slg:.3f}              {h_slg:.3f}
Hard-Hit%     {a_hard_hit:.1f}%             {h_hard_hit:.1f}%
최근5경기득점   {a_last5_scores}           {h_last5_scores}

───────────────────────────────────────
[불펜 & 팀 컨디션]
항목               {away_team}   {home_team}
불펜 ERA           {a_bullpen_era:.2f}        {h_bullpen_era:.2f}
최근5경기 승률      {a_roll:.1%}       {h_roll:.1%} ({momentum_home} / {momentum_away})
부상자 현황         {a_injuries}       {h_injuries}

───────────────────────────────────────
[SHAP — 모델이 승패를 가른 핵심 요인 Top 5]
{features_text}
(양수 = 홈팀 유리, 음수 = 원정팀 유리)

───────────────────────────────────────
[분석문 작성 규칙]

⟨구성 요건⟩
- 반드시 정확히 10문장 내외의 평문으로 작성 (마크다운, 이모지, 소제목 일절 금지)
- 각 문장은 서로 다른 분석 축(선발 구위 → 타선 득점력 → 불펜 → 흐름 → 구장 → 모델 신뢰도 순)을 다룰 것
- 문장 간에 "또한", "반면", "이를 고려하면", "결정적으로" 같은 접속어로 논리적 흐름 유지

⟨금지 사항⟩
- 단순 수치 복붙 나열 절대 금지 ("A팀 ERA는 X, B팀 ERA는 Y입니다" 식의 병렬 나열)
- "~입니다", "~있습니다" 식의 딱딱한 종결어 대신 분석가 어투("~가 유력하다", "~로 보인다", "~을 주목해야 한다") 사용
- 동일 표현·동일 구조 반복 금지 (각 문장이 독립적인 통찰을 담을 것)
- 예시 문장 그대로 사용 금지

⟨반드시 포함해야 할 분석 요소⟩
1. 선발 ERA vs FIP 괴리 해석 — 현재 성적이 실력인지 운인지 판별
2. K/9과 구속 데이터를 연결한 구위 평가 — 타자가 왜 어렵거나 쉬운지
3. 두 팀 타선의 Hard-Hit%와 OPS를 연결한 득점 생산력 전망
4. 최근 5경기 득점 흐름과 승률로 팀 사이클(상승/하락) 판단
5. 불펜 ERA 차이가 접전 시나리오에서 미치는 영향
6. 파크팩터가 오늘 경기 스타일(투수전 vs 타격전)에 미치는 영향
7. SHAP Top 요인 중 결정적인 1~2개를 수치와 함께 명시하며 모델 신뢰도 뒷받침
8. 부상자가 있다면 전력 공백이 어느 포지션에서 발생하는지 언급
9. 위 모든 요소를 종합한 최종 승부 전망과 그 이유
10. 모델 예측이 55% 미만이면 "백중세" 표현과 함께 변수 언급, 65% 이상이면 "우세" 표현 사용

⟨경기별 서사 다변화 장치⟩
- ERA-FIP 괴리가 0.5 이상인 선발이 있으면 반드시 "거품/저평가" 논거로 활용
- Hard-Hit% 차이가 5%p 이상이면 타선 체급 논쟁의 핵심으로 삼을 것
- 최근 5경기 득점 합계가 한 팀이 다른 팀의 1.5배 이상이면 "타격 사이클 절정/침묵" 표현 사용
- 양 선발 K/9 합산이 18 이상이면 투수전 프레임으로 서술
- 파크팩터가 1.10 이상이면 불펜 소모 가능성 언급 필수

지금 바로 분석문만 작성하세요. 다른 설명이나 인사는 일절 불필요합니다.
"""


async def generate_reasoning(
    home_team: str,
    away_team: str,
    home_win_prob: float,
    shap_top5: list[dict],
    home_starter: str | None = None,
    away_starter: str | None = None,
) -> str:
    from src.ml.reasoning.llm_client import get_llm_client

    client = get_llm_client()

    fav = home_team if home_win_prob >= 0.5 else away_team
    fav_prob = home_win_prob if home_win_prob >= 0.5 else 1 - home_win_prob

    home_starter_text = home_starter or "홈 선발투수"
    away_starter_text = away_starter or "원정 선발투수"

    feature_lines = []
    for idx, item in enumerate(shap_top5, start=1):
        name = item.get("feature", "")
        value = item.get("value", 0)
        shap = item.get("shap_value", 0)

        label = FEATURE_LABELS.get(name, name)
        direction = "홈팀 유리" if shap > 0 else "원정팀 유리"

        try:
            value_text = f"{float(value):.3f}"
        except (TypeError, ValueError):
            value_text = str(value)

        try:
            shap_text = f"{float(shap):+.4f}"
        except (TypeError, ValueError):
            shap_text = str(shap)

        feature_lines.append(
            f"{idx}. {label}: {value_text} / SHAP {shap_text} / {direction}"
        )

    features_text = "\n".join(feature_lines) if feature_lines else "제공된 SHAP 피처 없음"

    prompt = f"""당신은 메이저리그(MLB) 전문 퀀트 야구 분석가입니다.
아래의 모델 예측값, 선발투수 정보, SHAP 핵심 요인을 바탕으로 짧지만 완성도 높은 경기 분석문을 작성하세요.

[경기 기본 정보]
- 매치업: {away_team} (원정) vs {home_team} (홈)
- 선발투수: {away_starter_text} (원정) vs {home_starter_text} (홈)
- 모델 예측: {home_team} 승 {home_win_prob:.1%} / {away_team} 승 {1 - home_win_prob:.1%}
- 최종 탑독: {fav} ({fav_prob:.1%})

[SHAP 핵심 요인 Top 5]
{features_text}
- SHAP 양수는 홈팀에 유리한 요인입니다.
- SHAP 음수는 원정팀에 유리한 요인입니다.

[작성 규칙]
1. 반드시 정확히 4문장으로만 작성하세요.
2. 절대 문장을 중간에 끊지 말고, 마지막 문장까지 완결된 문장으로 끝내세요.
3. 첫 문장에는 매치업, 선발투수 이름, 모델 예측 확률을 자연스럽게 포함하세요.
4. 두 번째 문장에는 SHAP Top 5 중 가장 결정적인 1~2개 요인을 수치와 함께 해석하세요.
5. 세 번째 문장에는 선발 구위, 실점 억제력, 타선 득점 생산력, 최근 흐름 중 데이터에 드러난 핵심만 연결해서 설명하세요.
6. 네 번째 문장은 반드시 "{fav}의 승리가 유력하다" 또는 "{fav}에 우위가 있다"로 끝내세요.
7. 마크다운, 이모지, 소제목, 번호, 인사말은 절대 쓰지 마세요.
8. "~입니다", "~있습니다"보다 "~로 보인다", "~을 주목해야 한다", "~가 유력하다" 같은 분석가 어투를 사용하세요.

분석문만 작성하세요.
"""

    try:
        text = await client.complete(prompt)

        # 혹시 모델이 너무 길게 쓰더라도 DB/프론트에서 잘리지 않도록 마지막 안전장치
        text = " ".join(str(text).split())
        if len(text) > 900:
            text = text[:900].rsplit(" ", 1)[0].rstrip()
            if not text.endswith((".", "다", "요")):
                text += "."

        return text

    except Exception as e:
        from src.common.logger import get_logger

        get_logger(__name__).warning("LLM 근거 생성 실패: %s", e)

        return (
            f"{away_team} vs {home_team} 경기에서 {away_starter_text}와 {home_starter_text}의 선발 매치업을 기준으로 "
            f"모델은 {fav}의 승리를 {fav_prob:.1%} 확률로 예측한다. "
            f"SHAP 핵심 요인을 보면 {fav} 쪽에 승부를 기울이는 변수가 더 강하게 반영된 흐름으로 보인다. "
            f"모델 예측상 {fav}에 우위가 있다."
        )
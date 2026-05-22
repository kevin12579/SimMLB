"""오늘 경기 BBref 기반 추론 스크립트 (모델 v1과 동일 피처)"""
import asyncio
import io
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.path.insert(0, ".")

import joblib
import numpy as np
import pandas as pd
import shap
from datetime import date
from pathlib import Path
from sqlalchemy import text
from sqlalchemy.dialects.postgresql import insert

from src.db.session import get_session
from src.db.models.games import Game
from src.db.models.predictions import GamePrediction
from src.db.models.teams import Team
from src.collector.mlb_statsapi_client import MLBStatsAPIClient
from src.ml.features.bref_features import (
    load_bref_pitching, load_bref_batting, load_park_factors,
    build_feature_row, FEATURE_COLS,
)
from src.ml.reasoning.prompt_builder import generate_reasoning
from src.common.logger import get_logger

logger = get_logger("inference_v2")
MODEL_DIR = Path("models")


def _load_models():
    lgbm = joblib.load(MODEL_DIR / "lgbm_v1.pkl")
    xgb  = joblib.load(MODEL_DIR / "xgb_v1.pkl")
    cal  = joblib.load(MODEL_DIR / "calibrator_v1.pkl")
    return lgbm, xgb, cal


def _confidence(prob: float) -> str:
    diff = abs(prob - 0.5)
    if diff > 0.15:
        return "HIGH"
    if diff > 0.05:
        return "MED"
    return "LOW"


def _shap_top5(model, x_arr: np.ndarray) -> list[dict]:
    try:
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(x_arr)
        if isinstance(sv, list):
            sv = sv[1]
        row = sv[0]
        idx = np.argsort(np.abs(row))[::-1][:5]
        return [
            {"feature": FEATURE_COLS[i], "value": float(x_arr[0, i]), "shap_value": float(row[i])}
            for i in idx
        ]
    except Exception as e:
        logger.warning("SHAP 계산 실패: %s", e)
        return []


async def run(target_date: date | None = None) -> None:
    today = target_date or date.today()
    logger.info("=== 추론 시작: %s ===", today)

    # 1. 팀 동기화 → 경기 일정 동기화
    async with MLBStatsAPIClient() as client:
        with get_session() as session:
            await client.sync_teams(session)
        with get_session() as session:
            await client.sync_schedule(today, session)

    # 2. BBref 데이터 + 파크팩터 로드
    pitch_stats  = load_bref_pitching()
    bat_stats    = load_bref_batting()
    park_factors = load_park_factors()
    logger.info("BBref 로드 완료 — 투구:%d  타격:%d", len(pitch_stats), len(bat_stats))

    # 3. 모델 로드
    lgbm, xgb, cal_data = _load_models()
    w_lgbm = cal_data["weights"]["lgbm"]
    w_xgb  = cal_data["weights"]["xgb"]
    calibrator = cal_data["calibrator"]

    # 4. 팀 정보 + 롤링 승률
    with get_session() as session:
        teams = session.execute(text("SELECT mlbam_team_id, abbreviation FROM teams")).fetchall()
        id_to_abbr = {row[0]: row[1] for row in teams}

        # 롤링 승률: games 테이블 전체 (2023-2024 데이터 기반)
        all_games = session.execute(text("""
            SELECT home_team_id, away_team_id, home_score, away_score, game_date
            FROM games WHERE status = 'Final'
              AND home_score IS NOT NULL
            ORDER BY game_date
        """)).fetchall()

        team_results: dict[int, list] = {}
        for g in all_games:
            home_won = g.home_score > g.away_score
            for tid, won in [(g.home_team_id, home_won), (g.away_team_id, not home_won)]:
                team_results.setdefault(tid, []).append((g.game_date, won))

        def rolling_win(tid: int) -> float:
            past = team_results.get(tid, [])
            if not past:
                return 0.500
            recent = past[-20:]
            return sum(w for _, w in recent) / len(recent)

        # 5. 오늘 예정 경기
        today_games = session.query(Game).filter(
            Game.game_date == today,
            Game.status.notin_(["Final", "Postponed", "Cancelled"]),
        ).all()
        logger.info("오늘 경기: %d경기", len(today_games))

        if not today_games:
            logger.info("예정된 경기 없음")
            return

        count = 0
        for game in today_games:
            try:
                h_abbr = id_to_abbr.get(game.home_team_id, "")
                a_abbr = id_to_abbr.get(game.away_team_id, "")
                season = today.year

                feat = build_feature_row(
                    h_abbr=h_abbr,
                    a_abbr=a_abbr,
                    season=season,
                    h_roll=rolling_win(game.home_team_id),
                    a_roll=rolling_win(game.away_team_id),
                    pitch_stats=pitch_stats,
                    bat_stats=bat_stats,
                    park_factors=park_factors,
                )
                x_arr = np.array([[feat[c] for c in FEATURE_COLS]])

                lgbm_prob = float(lgbm.predict_proba(x_arr)[0, 1])
                xgb_prob  = float(xgb.predict_proba(x_arr)[0, 1])
                ensemble  = w_lgbm * lgbm_prob + w_xgb * xgb_prob
                home_win_prob = float(np.clip(calibrator.predict([ensemble])[0], 0.05, 0.95))

                shap_top5 = _shap_top5(lgbm, x_arr)
                reasoning = await generate_reasoning(h_abbr, a_abbr, home_win_prob, shap_top5)

                stmt = insert(GamePrediction).values(
                    game_pk=game.game_pk,
                    prediction_date=today,
                    home_win_prob=home_win_prob,
                    away_win_prob=round(1.0 - home_win_prob, 4),
                    confidence_level=_confidence(home_win_prob),
                    model_version="v1",
                    lgbm_prob=round(lgbm_prob, 4),
                    xgb_prob=round(xgb_prob, 4),
                    shap_top5=shap_top5,
                    reasoning_text=reasoning,
                ).on_conflict_do_update(
                    index_elements=["game_pk"],
                    set_=dict(
                        home_win_prob=home_win_prob,
                        away_win_prob=round(1.0 - home_win_prob, 4),
                        confidence_level=_confidence(home_win_prob),
                        shap_top5=shap_top5,
                        reasoning_text=reasoning,
                    ),
                )
                session.execute(stmt)
                count += 1
                logger.info(
                    "%s @ %s → 홈 승리 %.1f%%  [%s]",
                    a_abbr, h_abbr, home_win_prob * 100, _confidence(home_win_prob),
                )

            except Exception as e:
                logger.error("game_pk=%d 예측 실패: %s", game.game_pk, e)

        session.commit()
        logger.info("=== 추론 완료: %d경기 예측 저장 ===", count)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=str, default=None, help="YYYY-MM-DD (default: today)")
    args = parser.parse_args()
    target = date.fromisoformat(args.date) if args.date else None
    asyncio.run(run(target))
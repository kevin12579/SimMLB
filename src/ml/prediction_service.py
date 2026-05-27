"""ML 추론 파이프라인 — 19:30 KST 자동 실행"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from src.db.models.games import Game
from src.db.models.predictions import GamePrediction
from src.db.models.teams import Team
from src.ml.feature_engineering import build_feature_vector
from src.ml.models.ensemble import EnsembleCalibrator
from src.ml.reasoning.prompt_builder import generate_reasoning
from src.common.logger import get_logger

logger = get_logger(__name__)

MODEL_DIR = Path("models")


def _load_models() -> tuple:
    lgbm = joblib.load(MODEL_DIR / "lgbm_v1.pkl")
    xgb  = joblib.load(MODEL_DIR / "xgb_v1.pkl")
    cal  = EnsembleCalibrator.load(str(MODEL_DIR / "calibrator_v1.pkl"))
    return lgbm, xgb, cal


def _confidence(prob: float) -> str:
    diff = abs(prob - 0.5)
    if diff > 0.10:
        return "HIGH"
    if diff > 0.05:
        return "MED"
    return "LOW"


def _get_shap_top5(model: object, X: pd.DataFrame) -> list[dict]:
    try:
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X)
        if isinstance(sv, list):
            sv = sv[1]  # binary: 양성 클래스
        row = sv[0]
        indices = np.argsort(np.abs(row))[::-1][:5]
        return [
            {
                "feature":    X.columns[i],
                "value":      float(X.iloc[0, i]),
                "shap_value": float(row[i]),
            }
            for i in indices
        ]
    except Exception as e:
        logger.warning("SHAP 계산 실패: %s", e)
        return []


async def run_daily_predictions(today: date, session: Session) -> int:
    """오늘 경기 전체 예측 실행, 예측 생성 수 반환"""
    games = session.query(Game).filter(
        Game.game_date == today,
        Game.status.in_(["scheduled", "pre-game"]),
    ).all()

    if not games:
        logger.info("No scheduled games for %s", today)
        return 0

    try:
        lgbm, xgb, calibrator = _load_models()
    except FileNotFoundError:
        logger.error("모델 파일 없음. Week 3 모델 학습 후 재실행하세요.")
        return 0

    team_map: dict[int, str] = {
        t.mlbam_team_id: t.abbreviation
        for t in session.query(Team).all()
    }

    count = 0
    for game in games:
        try:
            features = build_feature_vector(game, session)
            X = pd.DataFrame([features])
            feature_cols = [c for c in X.columns if c not in ("game_pk", "game_date")]
            X_feat = X[feature_cols]

            lgbm_prob = float(lgbm.predict_proba(X_feat)[0, 1])
            xgb_prob  = float(xgb.predict_proba(X_feat)[0, 1])
            prob_map  = {"lgbm": np.array([lgbm_prob]), "xgb": np.array([xgb_prob])}
            home_win_prob = float(calibrator.predict(prob_map)[0])
            home_win_prob = max(0.05, min(0.95, home_win_prob))

            shap_top5 = _get_shap_top5(lgbm, X_feat)

            home_team = team_map.get(game.home_team_id, "HOME")
            away_team = team_map.get(game.away_team_id, "AWAY")
            reasoning = await generate_reasoning(home_team, away_team, home_win_prob, shap_top5)

            stmt = insert(GamePrediction).values(
                game_pk=game.game_pk,
                prediction_date=today,
                home_win_prob=home_win_prob,
                away_win_prob=1.0 - home_win_prob,
                confidence_level=_confidence(home_win_prob),
                model_version="v1",
                lgbm_prob=lgbm_prob,
                xgb_prob=xgb_prob,
                shap_top5=shap_top5,
                reasoning_text=reasoning,
            ).on_conflict_do_update(
                index_elements=["game_pk"],
                set_=dict(
                    home_win_prob=home_win_prob,
                    away_win_prob=1.0 - home_win_prob,
                    confidence_level=_confidence(home_win_prob),
                    shap_top5=shap_top5,
                    reasoning_text=reasoning,
                ),
            )
            session.execute(stmt)
            count += 1
            logger.info("Predicted %s @ %s → %.1f%%", away_team, home_team, home_win_prob * 100)

        except Exception as e:
            logger.error("Prediction failed for game %d: %s", game.game_pk, e)

    session.commit()
    logger.info("Daily predictions done: %d games", count)
    return count

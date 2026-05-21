"""Live Feed 통합 47피처 추론 — 경기별 호출 (T-15 동적 워커에서 game_pk 받음).

run_all_today(): 19:30 KST 폴백 또는 백필용 일괄 추론
run_single(pk): 단일 경기 추론
"""
from __future__ import annotations

import asyncio
import io
import sys
from datetime import date
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.path.insert(0, ".")

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import shap  # noqa: E402
from sqlalchemy import text  # noqa: E402
from sqlalchemy.dialects.postgresql import insert  # noqa: E402

from src.collector.live_feed_client import LiveFeedClient  # noqa: E402
from src.collector.mlb_statsapi_client import MLBStatsAPIClient  # noqa: E402
from src.common.logger import get_logger  # noqa: E402
from src.db.models.games import Game  # noqa: E402
from src.db.models.predictions import GamePrediction  # noqa: E402
from src.db.session import get_session  # noqa: E402
from src.ml.features.bref_features_v2 import (  # noqa: E402
    FEATURE_COLS_V2,
    build_feature_row_v2,
    get_rest_days,
    load_batter_individual_from_db,
    load_bref_batting_v2,
    load_bref_pitching,
    load_park_factors,
    load_pitcher_individual_from_db,
    load_team_bat_statcast_from_db,
)
from src.ml.reasoning.prompt_builder import generate_reasoning  # noqa: E402

logger = get_logger("inference_v3")
MODEL_DIR = Path("models")


def _confidence(prob: float) -> str:
    diff = abs(prob - 0.5)
    if diff > 0.15:
        return "HIGH"
    if diff > 0.05:
        return "MED"
    return "LOW"


def _shap_top5(model, x: np.ndarray) -> list[dict]:
    try:
        sv = shap.TreeExplainer(model).shap_values(x)
        if isinstance(sv, list):
            sv = sv[1]
        idx = np.argsort(np.abs(sv[0]))[::-1][:5]
        return [
            {"feature": FEATURE_COLS_V2[i],
             "value": float(x[0, i]),
             "shap_value": float(sv[0][i])}
            for i in idx
        ]
    except Exception as e:
        logger.warning("SHAP 계산 실패: %s", e)
        return []


def _load_models() -> tuple:
    lgbm = joblib.load(MODEL_DIR / "lgbm_v2.pkl")
    xgb = joblib.load(MODEL_DIR / "xgb_v2.pkl")
    cal = joblib.load(MODEL_DIR / "calibrator_v2.pkl")
    assert cal["feature_cols"] == FEATURE_COLS_V2, (
        f"Feature columns mismatch: model={len(cal['feature_cols'])} vs v3={len(FEATURE_COLS_V2)}"
    )
    return lgbm, xgb, cal


def _rolling_win_map(session) -> dict[int, float]:
    """모든 팀의 최근 20경기 승률 (Final 경기만)."""
    rows = session.execute(text("""
        SELECT home_team_id, away_team_id, home_score, away_score
        FROM games WHERE status='Final' AND home_score IS NOT NULL
        ORDER BY game_date
    """)).fetchall()
    team_wins: dict[int, list[bool]] = {}
    for g in rows:
        home_won = g.home_score > g.away_score
        team_wins.setdefault(g.home_team_id, []).append(home_won)
        team_wins.setdefault(g.away_team_id, []).append(not home_won)
    return {
        tid: (sum(wins[-20:]) / len(wins[-20:])) if wins else 0.5
        for tid, wins in team_wins.items()
    }


async def run_single(game_pk: int) -> None:
    """단일 경기 추론 — Live Feed sync → 47피처 → ensemble → SHAP → LLM → DB UPSERT."""
    today = date.today()
    logger.info("=== inference v3 시작: game %d ===", game_pk)

    # 1) Live Feed 동기화 (날씨/라인업/선발)
    async with LiveFeedClient() as lfc:
        with get_session() as session:
            snap = await lfc.sync_to_db(game_pk, session)

    # 2) 모델 + 정적 캐시
    lgbm, xgb_model, cal = _load_models()
    w_l, w_x = cal["weights"]["lgbm"], cal["weights"]["xgb"]
    calibrator = cal["calibrator"]

    pitch_team = load_bref_pitching()
    bat_team = load_bref_batting_v2()
    park = load_park_factors()

    # 3) DB 의존 캐시
    with get_session() as session:
        game = session.query(Game).filter(Game.game_pk == game_pk).first()
        if not game:
            logger.error("game %d not found", game_pk)
            return

        team_rows = session.execute(
            text("SELECT mlbam_team_id, abbreviation FROM teams")
        ).fetchall()
        id_to_abbr = {r[0]: r[1] for r in team_rows}
        h_abbr = id_to_abbr.get(game.home_team_id, "")
        a_abbr = id_to_abbr.get(game.away_team_id, "")

        roll_map = _rolling_win_map(session)
        sc_pitcher = load_pitcher_individual_from_db(session, today)
        sc_team_bat = load_team_bat_statcast_from_db(session, today)
        lineup_ids = snap["home_lineup_ids"] + snap["away_lineup_ids"]
        sc_batter = load_batter_individual_from_db(session, lineup_ids, today)
        rest_cache = {
            game.home_team_id: get_rest_days(session, game.home_team_id, today),
            game.away_team_id: get_rest_days(session, game.away_team_id, today),
        }

        feat = build_feature_row_v2(
            h_abbr=h_abbr, a_abbr=a_abbr, season=today.year,
            h_roll=roll_map.get(game.home_team_id, 0.5),
            a_roll=roll_map.get(game.away_team_id, 0.5),
            h_starter_id=snap["home_starter_id"],
            a_starter_id=snap["away_starter_id"],
            h_lineup_ids=snap["home_lineup_ids"],
            a_lineup_ids=snap["away_lineup_ids"],
            h_team_id=game.home_team_id, a_team_id=game.away_team_id,
            pitch_team=pitch_team, bat_team=bat_team, park=park,
            sc_pitcher_indiv=sc_pitcher, sc_team_bat=sc_team_bat,
            sc_batter_indiv=sc_batter, rest_cache=rest_cache,
        )
        x = np.array([[feat[c] for c in FEATURE_COLS_V2]])

        # 4) 앙상블 + Isotonic
        lp = float(lgbm.predict_proba(x)[0, 1])
        xp = float(xgb_model.predict_proba(x)[0, 1])
        ens = w_l * lp + w_x * xp
        cal_p = float(np.clip(calibrator.predict([ens])[0], 0.05, 0.95))

        # 5) SHAP + LLM 근거
        top5 = _shap_top5(lgbm, x)
        reasoning = await generate_reasoning(h_abbr, a_abbr, cal_p, top5)

        # 6) UPSERT
        stmt = insert(GamePrediction).values(
            game_pk=game_pk,
            prediction_date=today,
            home_win_prob=cal_p,
            away_win_prob=round(1.0 - cal_p, 4),
            confidence_level=_confidence(cal_p),
            model_version="v2",
            lgbm_prob=round(lp, 4),
            xgb_prob=round(xp, 4),
            shap_top5=top5,
            reasoning_text=reasoning,
        ).on_conflict_do_update(
            index_elements=["game_pk"],
            set_=dict(
                home_win_prob=cal_p,
                away_win_prob=round(1.0 - cal_p, 4),
                confidence_level=_confidence(cal_p),
                lgbm_prob=round(lp, 4),
                xgb_prob=round(xp, 4),
                shap_top5=top5,
                reasoning_text=reasoning,
                model_version="v2",
            ),
        )
        session.execute(stmt)
        session.commit()

    logger.info(
        "✅ game %d: %s @ %s → home %.1f%% [%s]",
        game_pk, a_abbr, h_abbr, cal_p * 100, _confidence(cal_p),
    )

    # 7) Redis 캐시 무효화
    _invalidate_today_cache()


def _invalidate_today_cache() -> None:
    try:
        import redis
        from config.settings import settings
        r = redis.from_url(f"redis://{settings.redis_host}:{settings.redis_port}")
        r.delete("predictions:today")
    except Exception as e:
        logger.debug("Redis cache invalidate skipped: %s", e)


async def run_all_today(target_date: date | None = None) -> None:
    """폴백 모드 — 19:30 KST 일괄 추론."""
    today = target_date or date.today()
    async with MLBStatsAPIClient() as client:
        with get_session() as session:
            await client.sync_schedule(today, session)
            pks = [
                g.game_pk
                for g in session.query(Game)
                .filter(
                    Game.game_date == today,
                    Game.status.notin_(["Final", "Postponed", "Cancelled"]),
                )
                .all()
            ]
    logger.info("일괄 추론 대상: %d경기", len(pks))
    for pk in pks:
        try:
            await run_single(pk)
        except Exception as e:
            logger.error("game %d 실패: %s", pk, e, exc_info=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--game-pk", type=int)
    parser.add_argument("--date", type=str, help="YYYY-MM-DD (run_all_today)")
    args = parser.parse_args()
    if args.game_pk:
        asyncio.run(run_single(args.game_pk))
    else:
        td = date.fromisoformat(args.date) if args.date else None
        asyncio.run(run_all_today(td))

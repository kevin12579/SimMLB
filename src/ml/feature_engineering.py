"""피처 엔지니어링 통합 — 단일 경기 피처 벡터 생성"""
from datetime import date, timedelta

from sqlalchemy.orm import Session

from src.ml.features.team_features import get_team_features
from src.ml.features.pitcher_features import get_pitcher_features
from src.ml.features.batter_features import get_batter_features
from src.ml.features.context_features import get_context_features
from src.db.models.games import Game

FEATURE_NAMES: list[str] = []  # build_training_data 실행 후 채워짐


def build_feature_vector(game: Game, session: Session) -> dict[str, float]:
    """경기 1개의 피처 벡터 생성 (as_of = game_date - 1일)"""
    as_of = game.game_date - timedelta(days=1)

    home_team = get_team_features(game.home_team_id, as_of, is_home=True,  session=session)
    away_team = get_team_features(game.away_team_id, as_of, is_home=False, session=session)

    home_sp = get_pitcher_features(game.home_starter_id, as_of, is_home=True,  session=session)
    away_sp = get_pitcher_features(game.away_starter_id, as_of, is_home=False, session=session)

    home_bat = get_batter_features(game.home_team_id, as_of, is_home=True,  session=session)
    away_bat = get_batter_features(game.away_team_id, as_of, is_home=False, session=session)

    ctx = get_context_features(game.game_pk, game.venue_id, session=session)

    # 복합 차이 피처
    rest_diff = home_team.get("home_rest_days", 0) - away_team.get("away_rest_days", 0)

    features = {
        **home_team,
        **away_team,
        **home_sp,
        **away_sp,
        **home_bat,
        **away_bat,
        **ctx,
        "rest_diff": float(rest_diff),
    }
    return features

"""학습 데이터셋 빌드 — 2023~2024 시즌 완료 경기"""
from __future__ import annotations

import pandas as pd
from sqlalchemy.orm import Session

from src.db.models.games import Game
from src.ml.feature_engineering import build_feature_vector
from src.common.logger import get_logger

logger = get_logger(__name__)


def build_training_dataset(start_season: int, end_season: int, session: Session) -> pd.DataFrame:
    """완료 경기 전체의 피처 + 타깃 DataFrame 생성"""
    games = (
        session.query(Game)
        .filter(
            Game.status == "Final",
            Game.season >= start_season,
            Game.season <= end_season,
            Game.home_score.isnot(None),
            Game.away_score.isnot(None),
        )
        .order_by(Game.game_date)
        .all()
    )
    logger.info("Building training data from %d games", len(games))

    rows = []
    for i, game in enumerate(games):
        try:
            features = build_feature_vector(game, session)
            features["game_pk"]    = game.game_pk
            features["game_date"]  = game.game_date
            features["target_home_win"] = int(game.home_score > game.away_score)
            rows.append(features)
        except Exception as e:
            logger.warning("Skipping game %d: %s", game.game_pk, e)

        if i % 500 == 0:
            logger.info("Progress: %d / %d", i, len(games))

    df = pd.DataFrame(rows)
    logger.info("Training dataset shape: %s", df.shape)
    return df


def chronological_split(df: pd.DataFrame, val_ratio: float = 0.15, test_ratio: float = 0.15):
    """시간순 70/15/15 분할"""
    df = df.sort_values("game_date").reset_index(drop=True)
    n = len(df)
    val_start  = int(n * (1 - val_ratio - test_ratio))
    test_start = int(n * (1 - test_ratio))
    train = df.iloc[:val_start]
    val   = df.iloc[val_start:test_start]
    test  = df.iloc[test_start:]
    logger.info("Split — train:%d  val:%d  test:%d", len(train), len(val), len(test))
    return train, val, test


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from src.db.session import get_session

    with get_session() as session:
        df = build_training_dataset(2023, 2024, session)
        out = "data/training_sets/training_set.parquet"
        df.to_parquet(out, index=False)
        print(f"저장 완료: {out}  shape={df.shape}")

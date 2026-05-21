from sqlalchemy import Column, Integer, Float, Date, String, TIMESTAMP, ForeignKey, func, Text, JSON

from src.db.base import Base


class GamePrediction(Base):
    __tablename__ = "game_predictions"

    id               = Column(Integer, primary_key=True, autoincrement=True)
    game_pk          = Column(Integer, ForeignKey("games.game_pk"), nullable=False, unique=True)
    prediction_date  = Column(Date,    nullable=False, index=True)
    home_win_prob    = Column(Float,   nullable=False)
    away_win_prob    = Column(Float,   nullable=False)
    confidence_level = Column(String(5), nullable=False)  # HIGH / MED / LOW
    model_version    = Column(String(20))
    lgbm_prob        = Column(Float)
    xgb_prob         = Column(Float)
    shap_top5        = Column(JSON)    # [{feature, value, shap_value}, ...]
    reasoning_text   = Column(Text)    # 한국어 분석 근거
    is_correct       = Column(Integer) # 1=맞음, 0=틀림, NULL=미정
    # v2: 라이브 + 메타
    weather_temp_f       = Column(Float)
    weather_condition    = Column(String(50))
    weather_wind         = Column(String(50))
    live_lineup_synced_at = Column(TIMESTAMP(timezone=True))
    live_home_win_prob   = Column(Float)
    live_status          = Column(String(30))
    live_current_inning  = Column(Integer)
    live_score_home      = Column(Integer)
    live_score_away      = Column(Integer)
    live_updated_at      = Column(TIMESTAMP(timezone=True))
    created_at       = Column(TIMESTAMP(timezone=True), server_default=func.now())
    updated_at       = Column(TIMESTAMP(timezone=True), server_default=func.now(), onupdate=func.now())

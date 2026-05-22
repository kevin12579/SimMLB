from sqlalchemy import Column, Integer, String, Float, Boolean, TIMESTAMP, func
from src.db.base import Base


class User(Base):
    __tablename__ = "users"
    id            = Column(Integer, primary_key=True, autoincrement=True)
    username      = Column(String(50), unique=True, nullable=False, index=True)
    email         = Column(String(120), unique=True, nullable=False, index=True)
    password_hash = Column(String(200), nullable=False)
    is_active     = Column(Boolean, default=True)
    created_at    = Column(TIMESTAMP(timezone=True), server_default=func.now())


class UserPick(Base):
    __tablename__ = "user_picks"
    id          = Column(Integer, primary_key=True, autoincrement=True)
    user_id     = Column(Integer, nullable=False, index=True)
    game_pk     = Column(Integer, nullable=False)
    pick_team   = Column(String(10), nullable=False)
    pick_prob   = Column(Float)
    confidence  = Column(String(5))
    is_correct  = Column(Integer)   # 1/0/NULL
    game_date   = Column(String(10))
    home_team   = Column(String(10))
    away_team   = Column(String(10))
    created_at  = Column(TIMESTAMP(timezone=True), server_default=func.now())
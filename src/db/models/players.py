from sqlalchemy import Column, Integer, String, Float, Date, Boolean, TIMESTAMP, ForeignKey, func

from src.db.base import Base


class Player(Base):
    __tablename__ = "players"

    mlbam_id        = Column(Integer, primary_key=True)
    full_name       = Column(String(100), nullable=False)
    position        = Column(String(5),   nullable=False)
    bats            = Column(String(1))   # L / R / S
    throws          = Column(String(1))   # L / R
    birth_date      = Column(Date)
    status          = Column(String(20),  nullable=False, default="active")
    current_team_id = Column(Integer, ForeignKey("teams.mlbam_team_id"))
    created_at      = Column(TIMESTAMP(timezone=True), server_default=func.now())
    updated_at      = Column(TIMESTAMP(timezone=True), server_default=func.now(), onupdate=func.now())


class PlayerSeasonStats(Base):
    __tablename__ = "player_season_stats"

    id          = Column(Integer, primary_key=True, autoincrement=True)
    player_id   = Column(Integer, ForeignKey("players.mlbam_id"), nullable=False)
    season      = Column(Integer, nullable=False)
    as_of_date  = Column(Date,    nullable=False)
    # Pitcher stats
    era         = Column(Float)
    fip         = Column(Float)
    xfip        = Column(Float)
    whip        = Column(Float)
    k_pct       = Column(Float)
    bb_pct      = Column(Float)
    hr9         = Column(Float)
    innings_pitched = Column(Float)
    # Batter stats
    wrc_plus    = Column(Float)
    ops         = Column(Float)
    babip       = Column(Float)
    created_at  = Column(TIMESTAMP(timezone=True), server_default=func.now())

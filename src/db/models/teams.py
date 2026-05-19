from sqlalchemy import Column, Integer, String, TIMESTAMP, func

from src.db.base import Base


class Team(Base):
    __tablename__ = "teams"

    mlbam_team_id = Column(Integer, primary_key=True)
    name          = Column(String(100), nullable=False)
    abbreviation  = Column(String(5),   nullable=False)
    league        = Column(String(5),   nullable=False)   # 'AL' or 'NL'
    division      = Column(String(20),  nullable=False)
    venue_id      = Column(Integer,     nullable=False)
    venue_name    = Column(String(100), nullable=False)
    created_at    = Column(TIMESTAMP(timezone=True), server_default=func.now())
    updated_at    = Column(TIMESTAMP(timezone=True), server_default=func.now(), onupdate=func.now())

from sqlalchemy import Column, Integer, String, Float, Date, Boolean, TIMESTAMP, ForeignKey, func, Text, JSON

from src.db.base import Base


class Game(Base):
    __tablename__ = "games"

    game_pk          = Column(Integer, primary_key=True)
    game_date        = Column(Date,    nullable=False, index=True)
    game_datetime    = Column(TIMESTAMP(timezone=True))
    home_team_id     = Column(Integer, ForeignKey("teams.mlbam_team_id"), nullable=False)
    away_team_id     = Column(Integer, ForeignKey("teams.mlbam_team_id"), nullable=False)
    home_score       = Column(Integer)
    away_score       = Column(Integer)
    status           = Column(String(20), nullable=False, default="scheduled")
    venue_id         = Column(Integer)
    venue_name       = Column(String(100))
    is_dome          = Column(Boolean, default=False)
    home_starter_id  = Column(Integer, ForeignKey("players.mlbam_id"))
    away_starter_id  = Column(Integer, ForeignKey("players.mlbam_id"))
    season           = Column(Integer, nullable=False)
    created_at       = Column(TIMESTAMP(timezone=True), server_default=func.now())
    updated_at       = Column(TIMESTAMP(timezone=True), server_default=func.now(), onupdate=func.now())


class TeamDailySnapshot(Base):
    __tablename__ = "team_daily_snapshots"

    id              = Column(Integer, primary_key=True, autoincrement=True)
    team_id         = Column(Integer, ForeignKey("teams.mlbam_team_id"), nullable=False)
    snapshot_date   = Column(Date, nullable=False)
    season          = Column(Integer, nullable=False)
    wins            = Column(Integer, default=0)
    losses          = Column(Integer, default=0)
    win_rate        = Column(Float)
    last10_wins     = Column(Integer, default=0)
    last10_losses   = Column(Integer, default=0)
    last10_win_rate = Column(Float)
    home_wins       = Column(Integer, default=0)
    home_losses     = Column(Integer, default=0)
    runs_scored     = Column(Integer, default=0)
    runs_allowed    = Column(Integer, default=0)
    pythagenpat_wp  = Column(Float)
    streak_signed   = Column(Integer, default=0)  # 연승(+)/연패(-) 값
    rest_days       = Column(Integer, default=0)
    created_at      = Column(TIMESTAMP(timezone=True), server_default=func.now())


class GameLineup(Base):
    __tablename__ = "game_lineups"

    id          = Column(Integer, primary_key=True, autoincrement=True)
    game_pk     = Column(Integer, ForeignKey("games.game_pk"), nullable=False)
    game_date   = Column(Date, nullable=False, index=True)
    player_id   = Column(Integer, ForeignKey("players.mlbam_id"), nullable=False)
    team_id     = Column(Integer, ForeignKey("teams.mlbam_team_id"), nullable=False)
    batting_order = Column(Integer)
    position    = Column(String(5))
    is_home     = Column(Boolean, nullable=False)
    created_at  = Column(TIMESTAMP(timezone=True), server_default=func.now())


class PitcherGameLog(Base):
    __tablename__ = "pitcher_game_logs"

    id          = Column(Integer, primary_key=True, autoincrement=True)
    player_id   = Column(Integer, ForeignKey("players.mlbam_id"), nullable=False)
    game_pk     = Column(Integer, ForeignKey("games.game_pk"), nullable=False)
    team_id     = Column(Integer, ForeignKey("teams.mlbam_team_id"), nullable=False)
    game_date   = Column(Date, nullable=False, index=True)
    season      = Column(Integer, nullable=False)
    is_starter  = Column(Boolean, default=True)
    ip          = Column(Float)   # innings pitched
    er          = Column(Integer) # earned runs
    k           = Column(Integer) # strikeouts
    bb          = Column(Integer) # walks
    h           = Column(Integer) # hits
    hr          = Column(Integer) # home runs
    pitches     = Column(Integer)
    created_at  = Column(TIMESTAMP(timezone=True), server_default=func.now())


class BatterGameLog(Base):
    __tablename__ = "batter_game_logs"

    id          = Column(Integer, primary_key=True, autoincrement=True)
    player_id   = Column(Integer, ForeignKey("players.mlbam_id"), nullable=False)
    game_pk     = Column(Integer, ForeignKey("games.game_pk"), nullable=False)
    team_id     = Column(Integer, ForeignKey("teams.mlbam_team_id"), nullable=False)
    game_date   = Column(Date, nullable=False, index=True)
    season      = Column(Integer, nullable=False)
    ab          = Column(Integer)  # at bats
    h           = Column(Integer)  # hits
    hr          = Column(Integer)  # home runs
    rbi         = Column(Integer)
    bb          = Column(Integer)
    k           = Column(Integer)
    created_at  = Column(TIMESTAMP(timezone=True), server_default=func.now())


class StatcastPitch(Base):
    __tablename__ = "statcast_pitches"

    id               = Column(Integer, primary_key=True, autoincrement=True)
    game_date        = Column(Date, nullable=False, index=True)
    game_pk          = Column(Integer, ForeignKey("games.game_pk"))
    pitcher_id       = Column(Integer, ForeignKey("players.mlbam_id"))
    batter_id        = Column(Integer, ForeignKey("players.mlbam_id"))
    pitcher_team_id  = Column(Integer, ForeignKey("teams.mlbam_team_id"))
    batter_team_id   = Column(Integer, ForeignKey("teams.mlbam_team_id"))
    pitch_type       = Column(String(5))
    release_speed    = Column(Float)
    spin_rate        = Column(Float)
    launch_speed     = Column(Float)
    launch_angle     = Column(Float)
    is_hard_hit      = Column(Boolean, default=False)  # 95mph+
    is_barrel        = Column(Boolean, default=False)
    xwoba            = Column(Float)
    events           = Column(String(50))
    description      = Column(String(50))
    swing            = Column(Boolean, default=False)
    whiff            = Column(Boolean, default=False)
    created_at       = Column(TIMESTAMP(timezone=True), server_default=func.now())


class PlayerStatcastSummary(Base):
    __tablename__ = "player_statcast_summary"

    id              = Column(Integer, primary_key=True, autoincrement=True)
    player_id       = Column(Integer, ForeignKey("players.mlbam_id"), nullable=False)
    as_of_date      = Column(Date, nullable=False)
    season          = Column(Integer, nullable=False)
    xwoba           = Column(Float)
    barrel_rate     = Column(Float)
    hard_hit_pct    = Column(Float)
    chase_rate      = Column(Float)
    swstr_pct       = Column(Float)  # swinging strike %
    created_at      = Column(TIMESTAMP(timezone=True), server_default=func.now())


class GameWeather(Base):
    __tablename__ = "game_weather"

    id           = Column(Integer, primary_key=True, autoincrement=True)
    game_pk      = Column(Integer, ForeignKey("games.game_pk"), nullable=False, unique=True)
    temp_f       = Column(Float)
    wind_speed_mph = Column(Float)
    wind_dir     = Column(String(10))
    precip_mm    = Column(Float)
    created_at   = Column(TIMESTAMP(timezone=True), server_default=func.now())

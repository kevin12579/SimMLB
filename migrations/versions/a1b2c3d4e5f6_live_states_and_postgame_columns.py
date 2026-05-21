"""live_states_and_postgame_columns

Revision ID: a1b2c3d4e5f6
Revises: 4fb0b416e598
Create Date: 2026-05-21 14:00:00.000000

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op


revision: str = "a1b2c3d4e5f6"
down_revision: Union[str, Sequence[str], None] = "4fb0b416e598"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # game_lineups 인덱스 (game_pk + batting_order)
    op.create_index(
        "idx_game_lineups_pk_order",
        "game_lineups",
        ["game_pk", "batting_order"],
    )

    # game_predictions: live + 메타 컬럼 추가
    op.add_column("game_predictions", sa.Column("live_lineup_synced_at", sa.TIMESTAMP(timezone=True)))
    op.add_column("game_predictions", sa.Column("weather_temp_f", sa.Float()))
    op.add_column("game_predictions", sa.Column("weather_condition", sa.String(length=50)))
    op.add_column("game_predictions", sa.Column("weather_wind", sa.String(length=50)))
    op.add_column("game_predictions", sa.Column("live_home_win_prob", sa.Float()))
    op.add_column("game_predictions", sa.Column("live_status", sa.String(length=30)))
    op.add_column("game_predictions", sa.Column("live_current_inning", sa.Integer()))
    op.add_column("game_predictions", sa.Column("live_score_home", sa.Integer()))
    op.add_column("game_predictions", sa.Column("live_score_away", sa.Integer()))
    op.add_column("game_predictions", sa.Column("live_updated_at", sa.TIMESTAMP(timezone=True)))

    # 신규 테이블 game_live_states (1분 polling 시계열)
    op.create_table(
        "game_live_states",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("game_pk", sa.Integer(), sa.ForeignKey("games.game_pk"), nullable=False),
        sa.Column("polled_at", sa.TIMESTAMP(timezone=True), nullable=False,
                  server_default=sa.func.now()),
        sa.Column("game_status", sa.String(length=30), nullable=False),
        sa.Column("current_inning", sa.Integer()),
        sa.Column("inning_half", sa.String(length=10)),
        sa.Column("outs", sa.Integer()),
        sa.Column("balls", sa.Integer()),
        sa.Column("strikes", sa.Integer()),
        sa.Column("home_score", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("away_score", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("on_first", sa.Boolean(), server_default=sa.false()),
        sa.Column("on_second", sa.Boolean(), server_default=sa.false()),
        sa.Column("on_third", sa.Boolean(), server_default=sa.false()),
        sa.Column("mlb_win_prob", sa.Float()),
        sa.Column("live_home_prob", sa.Float(), nullable=False),
        sa.Column("created_at", sa.TIMESTAMP(timezone=True), server_default=sa.func.now()),
    )
    op.create_index(
        "idx_live_states_pk_time",
        "game_live_states",
        ["game_pk", sa.text("polled_at DESC")],
    )

    # pitcher_game_logs / batter_game_logs UNIQUE (player_id, game_pk) — postgame UPSERT용
    op.create_unique_constraint(
        "uq_pitcher_game_logs_pid_pk",
        "pitcher_game_logs",
        ["player_id", "game_pk"],
    )
    op.create_unique_constraint(
        "uq_batter_game_logs_pid_pk",
        "batter_game_logs",
        ["player_id", "game_pk"],
    )


def downgrade() -> None:
    op.drop_constraint("uq_batter_game_logs_pid_pk", "batter_game_logs", type_="unique")
    op.drop_constraint("uq_pitcher_game_logs_pid_pk", "pitcher_game_logs", type_="unique")
    op.drop_index("idx_live_states_pk_time", table_name="game_live_states")
    op.drop_table("game_live_states")
    for col in (
        "live_updated_at", "live_score_away", "live_score_home",
        "live_current_inning", "live_status", "live_home_win_prob",
        "weather_wind", "weather_condition", "weather_temp_f",
        "live_lineup_synced_at",
    ):
        op.drop_column("game_predictions", col)
    op.drop_index("idx_game_lineups_pk_order", table_name="game_lineups")

import os
from logging.config import fileConfig

from sqlalchemy import engine_from_config, pool
from alembic import context

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# 모든 모델 import (autogenerate 지원)
from src.db.base import Base  # noqa: E402
import src.db.models.teams       # noqa: E402, F401
import src.db.models.players     # noqa: E402, F401
import src.db.models.games       # noqa: E402, F401
import src.db.models.predictions # noqa: E402, F401

target_metadata = Base.metadata

# .env에서 DB URL 동적으로 구성
def get_url() -> str:
    from config.settings import settings
    return settings.database_url

config.set_main_option("sqlalchemy.url", get_url())


def run_migrations_offline() -> None:
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()

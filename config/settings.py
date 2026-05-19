from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "mlb_prediction"
    postgres_user: str = "mlb_user"
    postgres_password: str = ""
    redis_host: str = "localhost"
    redis_port: int = 6379
    llm_provider: str = "openai"
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    groq_api_key: str = ""
    groq_model: str = "llama-3.1-8b-instant"
    discord_webhook_url: str = ""
    sentry_dsn: str = ""
    log_level: str = "INFO"
    environment: str = "development"

    @property
    def database_url(self) -> str:
        return (
            f"postgresql+psycopg2://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()

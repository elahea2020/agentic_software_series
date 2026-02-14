"""
Application settings loaded from environment variables and .env file.

Uses pydantic-settings to automatically read the `.env` file at the project
root and expose typed configuration via a singleton ``settings`` instance.

Usage
-----
    from core.settings import settings

    print(settings.anthropic_api_key)
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
    )

    anthropic_api_key: str


settings = Settings()
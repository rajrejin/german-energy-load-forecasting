"""API settings (environment variables, prefix ENERGY_API_)."""

from __future__ import annotations

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ApiSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="ENERGY_API_", env_file=".env", extra="ignore", populate_by_name=True
    )

    api_key: str | None = Field(
        default=None, validation_alias=AliasChoices("ENERGY_API_KEY", "api_key")
    )  # when set, requests need the X-API-Key header
    cors_origins: str = ""  # comma separated; empty means CORS is off
    request_timeout_s: float = 10.0
    json_logs: bool = True
    max_forecast_age_days: int = 400

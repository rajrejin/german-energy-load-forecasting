"""Environment-driven settings shared by the refresh job and the API."""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict

from energy.store.base import ArtifactStore, make_store


class StoreSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ENERGY_", env_file=".env", extra="ignore")

    store: str = "local"  # local | azurite | azure
    store_root: str = "artifacts/store"
    container: str = "energy"
    azure_connection_string: str | None = None
    azure_account_url: str | None = None
    stale_after_hours: float = 36.0


def store_from_env(kind: str | None = None) -> ArtifactStore:
    s = StoreSettings()
    k = kind or s.store
    if k == "local":
        return make_store("local", root=s.store_root)
    if k == "azurite":
        return make_store("azurite", container=s.container)
    return make_store(
        "azure",
        container=s.container,
        connection_string=s.azure_connection_string,
        account_url=s.azure_account_url,
    )

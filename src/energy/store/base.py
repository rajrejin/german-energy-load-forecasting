"""One artifact store interface with local filesystem, Azurite and Azure Blob implementations."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class ArtifactStore(ABC):
    """Key/value store for model artifacts and forecast outputs. Keys use forward slashes."""

    @abstractmethod
    def put_bytes(self, key: str, data: bytes) -> None: ...

    @abstractmethod
    def get_bytes(self, key: str) -> bytes: ...

    @abstractmethod
    def exists(self, key: str) -> bool: ...

    @abstractmethod
    def list_keys(self, prefix: str = "") -> list[str]: ...

    @abstractmethod
    def delete(self, key: str) -> None: ...

    # -- helpers -------------------------------------------------------------------------
    def put_json(self, key: str, obj: Any) -> None:
        self.put_bytes(key, json.dumps(obj, indent=1).encode("utf-8"))

    def get_json(self, key: str) -> Any:
        return json.loads(self.get_bytes(key).decode("utf-8"))

    def upload_dir(self, local: Path, prefix: str) -> list[str]:
        keys = []
        for p in sorted(local.rglob("*")):
            if p.is_file():
                key = f"{prefix}/{p.relative_to(local).as_posix()}"
                self.put_bytes(key, p.read_bytes())
                keys.append(key)
        return keys

    def download_dir(self, prefix: str, local: Path) -> list[Path]:
        out = []
        for key in self.list_keys(prefix):
            rel = key[len(prefix) :].lstrip("/")
            dest = local / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(self.get_bytes(key))
            out.append(dest)
        return out


class KeyNotFoundError(KeyError):
    pass


def make_store(kind: str = "local", **kw: Any) -> ArtifactStore:
    """kind: 'local' (root=...), 'azurite' (connection string for the emulator), or 'azure'
    (connection_string or account_url with DefaultAzureCredential)."""
    if kind == "local":
        from energy.store.local import LocalStore

        return LocalStore(Path(kw.get("root", "artifacts/store")))
    from energy.store.azure_blob import AzureBlobStore

    if kind == "azurite":
        return AzureBlobStore.azurite(container=kw.get("container", "energy"))
    if kind == "azure":
        return AzureBlobStore(
            container=kw.get("container", "energy"),
            connection_string=kw.get("connection_string"),
            account_url=kw.get("account_url"),
        )
    raise ValueError(f"unknown store kind {kind!r}")

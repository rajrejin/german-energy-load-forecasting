"""Local filesystem store."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from energy.store.base import ArtifactStore, KeyNotFoundError


class LocalStore(ArtifactStore):
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _p(self, key: str) -> Path:
        p = (self.root / key).resolve()
        if self.root.resolve() not in p.parents and p != self.root.resolve():
            raise ValueError(f"key escapes the store root: {key}")
        return p

    def put_bytes(self, key: str, data: bytes) -> None:
        p = self._p(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        # atomic replace so readers never see a partial file
        fd, tmp = tempfile.mkstemp(dir=p.parent)
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.replace(tmp, p)

    def get_bytes(self, key: str) -> bytes:
        p = self._p(key)
        if not p.is_file():
            raise KeyNotFoundError(key)
        return p.read_bytes()

    def exists(self, key: str) -> bool:
        return self._p(key).is_file()

    def list_keys(self, prefix: str = "") -> list[str]:
        base = self.root.resolve()
        return sorted(
            p.relative_to(base).as_posix()
            for p in base.rglob("*")
            if p.is_file() and p.relative_to(base).as_posix().startswith(prefix)
        )

    def delete(self, key: str) -> None:
        p = self._p(key)
        if p.is_file():
            p.unlink()

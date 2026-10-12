"""Artifact manifests: reproducibility metadata that the API checks before serving."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

from energy.config import Config

REQUIRED_KEYS = (
    "model",
    "zone",
    "config_hash",
    "data_hash",
    "seed",
    "library_versions",
    "train_period",
    "git_commit",
    "val_metrics",
    "created_utc",
    "files",
)


def file_hash(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()[:16]


def git_commit(root: Path) -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return out.stdout.strip() or "unknown"
    except Exception:  # pragma: no cover
        return "unknown"


def library_versions() -> dict[str, str]:
    out = {"python": platform.python_version()}
    for lib in ("numpy", "pandas", "torch", "lightgbm", "statsforecast", "shap", "scikit-learn"):
        try:
            out[lib] = metadata.version(lib)
        except metadata.PackageNotFoundError:  # pragma: no cover
            out[lib] = "missing"
    return out


def write_manifest(
    folder: Path,
    *,
    model: str,
    zone: str,
    cfg: Config,
    data_path: Path,
    seed: int,
    train_period: tuple[str, str],
    val_metrics: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    files = {
        p.name: file_hash(p)
        for p in sorted(folder.iterdir())
        if p.is_file() and p.name != "manifest.json"
    }
    m: dict[str, Any] = {
        "model": model,
        "zone": zone,
        "config_hash": cfg.hash(),
        "data_hash": file_hash(data_path) if data_path.exists() else "n/a",
        "seed": seed,
        "library_versions": library_versions(),
        "train_period": list(train_period),
        "git_commit": git_commit(cfg.root),
        "val_metrics": val_metrics,
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "files": files,
        **(extra or {}),
    }
    (folder / "manifest.json").write_text(json.dumps(m, indent=2))
    return m


def validate_manifest(folder: Path) -> dict[str, Any]:
    """Raise ValueError when the manifest is missing, incomplete or inconsistent with files."""
    p = folder / "manifest.json"
    if not p.exists():
        raise ValueError(f"manifest missing in {folder}")
    m: dict[str, Any] = json.loads(p.read_text())
    missing = [k for k in REQUIRED_KEYS if k not in m]
    if missing:
        raise ValueError(f"manifest incomplete, missing {missing}")
    for name, h in m["files"].items():
        f = folder / name
        if not f.exists():
            raise ValueError(f"file {name} listed in manifest is absent")
        if file_hash(f) != h:
            raise ValueError(f"file {name} hash differs from manifest")
    return m

"""Typed configuration loaded from configs/default.yaml and configs/zones.yaml."""

from __future__ import annotations

import hashlib
import json
import os
import random
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from pydantic import BaseModel, Field

ZONES = ["50Hertz", "Amprion", "TenneT", "TransnetBW"]
ALL_REGIONS = ["DE", *ZONES]


def project_root() -> Path:
    env = os.environ.get("ENERGY_ROOT")
    if env:
        return Path(env)
    here = Path(__file__).resolve()
    for p in [Path.cwd(), *here.parents]:
        if (p / "configs" / "default.yaml").exists():
            return p
    return Path.cwd()


class Splits(BaseModel):
    train_end: str
    val_end: str
    test_start: str


class Paths(BaseModel):
    raw: str = "data/raw"
    processed: str = "data/processed"
    artifacts: str = "artifacts"
    results: str = "results"
    logs: str = "logs"


class SmardCfg(BaseModel):
    base_url: str
    delay_s: float = 0.15
    refetch_recent_days: int = 14


class OpenMeteoCfg(BaseModel):
    archive_url: str
    hist_forecast_url: str
    forecast_url: str
    delay_s: float = 1.0
    variables: list[str]


class QualityCfg(BaseModel):
    max_interp_gap_h: int = 3
    outlier_z: float = 8.0


class FeaturesCfg(BaseModel):
    hdd_base_c: float = 15.0
    cdd_base_c: float = 22.0


class LstmCfg(BaseModel):
    lookback: int = 168
    hidden: int = 64
    layers: int = 1
    dropout: float = 0.1
    lr: float = 0.002
    weight_decay: float = 1e-4
    batch_size: int = 256
    max_epochs: int = 40
    patience: int = 5
    clip: float = 1.0
    quantiles: list[float] = Field(default_factory=lambda: [0.1, 0.5, 0.9])
    stride_train: int = 1


class GbmCfg(BaseModel):
    n_estimators: int = 600
    learning_rate: float = 0.05
    num_leaves: int = 63
    min_child_samples: int = 40
    subsample: float = 0.8
    colsample_bytree: float = 0.8


class ParallelCfg(BaseModel):
    workers: int = 4
    threads_per_worker: int = 1
    memory_limit: str = "1100MB"


class Config(BaseModel):
    seed: int = 0
    timezone: str = "Europe/Berlin"
    issue_hour_local: int = 10
    data_start: str = "2015-01-01"
    splits: Splits
    paths: Paths = Paths()
    smard: SmardCfg
    openmeteo: OpenMeteoCfg
    quality: QualityCfg = QualityCfg()
    features: FeaturesCfg = FeaturesCfg()
    lstm: LstmCfg = LstmCfg()
    gbm: GbmCfg = GbmCfg()
    parallel: ParallelCfg = ParallelCfg()
    root: Path = Field(default_factory=project_root)

    def path(self, name: str) -> Path:
        p = self.root / getattr(self.paths, name)
        p.mkdir(parents=True, exist_ok=True)
        return p

    def hash(self) -> str:
        d = self.model_dump(mode="json", exclude={"root"})
        return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()[:12]


@lru_cache(maxsize=4)
def load_config(path: str | None = None) -> Config:
    root = project_root()
    p = Path(path) if path else root / "configs" / "default.yaml"
    with open(p, encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f)
    return Config(**raw, root=root)


@lru_cache(maxsize=2)
def load_zones(path: str | None = None) -> dict[str, Any]:
    p = Path(path) if path else project_root() / "configs" / "zones.yaml"
    with open(p, encoding="utf-8") as f:
        out: dict[str, Any] = yaml.safe_load(f)
    return out


def set_seed(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:  # pragma: no cover
        pass

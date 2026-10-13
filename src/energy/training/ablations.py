"""Ablation models (validation-based early stopping only; test numbers are reported once)."""

from __future__ import annotations

import logging
from typing import Any

from energy.config import ZONES, Config, load_config
from energy.evaluation.backtest import make_context
from energy.features import GBM_FEATURES
from energy.models.pooled_lstm import PooledLSTM
from energy.training.manifest import write_manifest
from energy.training.train import artifact_dir, fit_model, load_best_hp, save_artifact

log = logging.getLogger(__name__)

WEATHER_FEATURES = [f for f in GBM_FEATURES if f.startswith(("w_", "temp_")) or f in ("hdd", "cdd")]
GEN_LAGS = [f"{g}_lag{lag}" for g in ("gen_wind", "gen_pv", "gen_fossil") for lag in (48, 168)]

ABLATIONS: dict[str, tuple[str, dict[str, Any]]] = {
    "lstm_abl_noweather": ("lstm", {"use_weather": False}),
    "lstm_abl_nocalemb": ("lstm", {"use_cal_emb": False}),
    "lstm_abl_lb72": ("lstm", {"lookback": 72}),
    "lstm_abl_lb336": ("lstm", {"lookback": 336}),
    "lstm_abl_no2020": ("lstm", {"train_exclude": [("2020-01-01", "2020-12-31")]}),
    "lstm_abl_no2022": ("lstm", {"train_exclude": [("2022-01-01", "2022-12-31")]}),
    "lightgbm_abl_noweather": (
        "lightgbm",
        {"features": [f for f in GBM_FEATURES if f not in WEATHER_FEATURES]},
    ),
    "lightgbm_abl_laggen": ("lightgbm", {"features": [*GBM_FEATURES, *GEN_LAGS]}),
}


def run_ablations(
    zone: str = "DE", names: list[str] | None = None, cfg: Config | None = None
) -> None:
    cfg = cfg or load_config()
    ctx = make_context(zone, cfg)
    for name in names or list(ABLATIONS):
        if (artifact_dir(cfg, zone, name) / "manifest.json").exists():
            log.info("skip %s (exists)", name)
            continue
        kind, opts = ABLATIONS[name]
        model, met = fit_model(kind, ctx, seed=0, **opts)
        save_artifact(name, ctx, model, met, 0)


def finalize_pooled(folder, cfg: Config) -> None:
    """Write the manifest of a saved pooled model (validation loss read from its history)."""
    import json

    from energy.data.build import processed_path
    from energy.evaluation.backtest import make_context as _mc

    meta = json.loads((folder / "meta.json").read_text())
    best = min(h["val_loss"] for h in meta["history"])
    first = meta["zones"][0]
    dates = _mc(first, cfg).dates
    write_manifest(
        folder,
        model="pooled_lstm",
        zone="pooled:" + ",".join(meta["zones"]),
        cfg=cfg,
        data_path=processed_path(first, cfg),
        seed=meta["seed"],
        train_period=(dates["train"][0], dates["train"][1]),
        val_metrics={"val_pinball_scaled": float(best)},
    )


def train_pooled(cfg: Config | None = None, hp: dict[str, Any] | None = None) -> None:
    cfg = cfg or load_config()
    folder = cfg.path("artifacts") / "models" / "pooled" / "pooled_lstm"
    if (folder / "manifest.json").exists():
        log.info("pooled model exists")
        return
    if not (folder / "model.pt").exists():
        ctxs = {z: make_context(z, cfg) for z in ZONES}
        hp = hp or load_best_hp("lstm", cfg)
        p = PooledLSTM(cfg, ZONES, seed=0, hp=hp)
        p.fit(ctxs)
        folder.mkdir(parents=True, exist_ok=True)
        p.save(folder)
    finalize_pooled(folder, cfg)

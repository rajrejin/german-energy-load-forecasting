"""Backtest runner: loads artifacts, writes per-model predictions, applies CQR calibration."""

from __future__ import annotations

import json
import logging

import numpy as np

from energy.config import ALL_REGIONS, Config, load_config
from energy.evaluation.backtest import (
    Context,
    Preds,
    collect,
    load_preds,
    make_context,
    pred_path,
    save_preds,
)
from energy.evaluation.intervals import apply_offset, cqr_offset
from energy.models.base import Forecaster
from energy.models.stat_mstl import MSTLARIMA
from energy.training.train import artifact_dir, load_artifact

log = logging.getLogger(__name__)

WEATHERLESS = {"naive_week", "naive_day", "mstl_arima"}


def base_kind(name: str) -> str:
    for k in ("naive_week", "naive_day", "lightgbm", "lstm", "pooled"):
        if name.startswith(k):
            return k
    raise ValueError(name)


def calib_path(cfg: Config):
    return cfg.path("results") / "calibration.json"


def read_calibration(cfg: Config) -> dict:
    p = calib_path(cfg)
    return json.loads(p.read_text()) if p.exists() else {}


def update_calibration(cfg: Config, zone: str, name: str, regime: str, q: float) -> None:
    d = read_calibration(cfg)
    d.setdefault(zone, {}).setdefault(name, {})[regime] = q
    calib_path(cfg).write_text(json.dumps(d, indent=2))


def with_calibration(pr: Preds, q_off: float | None) -> Preds:
    if q_off is not None and 0.1 in pr.q and 0.9 in pr.q:
        lo, hi = apply_offset(pr.q[0.1], pr.q[0.9], q_off)
        pr.q[0.1], pr.q[0.9] = lo, hi
    return pr


def get_preds(
    cfg: Config, regime: str, zone: str, split: str, name: str, calibrated: bool = True
) -> Preds:
    """Load stored predictions; weather-free models are stored once under 'reanalysis'."""
    reg = "reanalysis" if name in WEATHERLESS else regime
    pr = load_preds(pred_path(cfg, reg, zone, split, name))
    if calibrated:
        c = read_calibration(cfg).get(zone, {}).get(name, {})
        pr = with_calibration(pr, c.get(reg))
    return pr


def run_backtest(
    zone: str,
    names: list[str],
    splits: tuple[str, ...] = ("val",),
    regimes: tuple[str, ...] = ("reanalysis", "archived"),
    cfg: Config | None = None,
    force: bool = False,
    ctx: Context | None = None,
) -> None:
    cfg = cfg or load_config()
    ctx = ctx or make_context(zone, cfg)
    for name in names:
        if name == "mstl_arima":
            _run_mstl(ctx, splits, force)
            continue
        kind = base_kind(name)
        model: Forecaster
        if name == "pooled_lstm":
            from energy.models.pooled_lstm import PooledLSTM

            pooled = PooledLSTM.load(
                cfg.path("artifacts") / "models" / "pooled" / "pooled_lstm", cfg
            )
            if zone not in pooled.zones:
                continue
            model = pooled.view(zone)
        else:
            model = load_artifact(kind, zone, cfg, name)
        for regime in regimes:
            if name in WEATHERLESS and regime != "reanalysis":
                continue
            for split in ("val", "test"):
                if split not in splits:
                    continue
                path = pred_path(cfg, regime, zone, split, name)
                if path.exists() and not force:
                    continue
                o = ctx.origins(split, regime)
                if not o.valid.any():
                    continue
                pr = collect(model, ctx, o)
                save_preds(path, pr)
                if split == "val" and 0.1 in pr.q:
                    q = cqr_offset(pr.y, pr.q[0.1], pr.q[0.9])
                    update_calibration(cfg, zone, name, regime, q)
                log.info("%s %s %s %s: %d origins", zone, name, regime, split, len(pr.y))


def _run_mstl(ctx: Context, splits: tuple[str, ...], force: bool) -> None:
    cfg = ctx.cfg
    model = MSTLARIMA()
    for split in ("val", "test"):
        if split not in splits:
            continue
        path = pred_path(cfg, "reanalysis", ctx.zone, split, "mstl_arima")
        if path.exists() and not force:
            continue
        o = ctx.origins(split, "reanalysis")
        p = model.predict(ctx.zf, o)
        m = o.valid & ~np.isnan(p).any(axis=1)
        pr = Preds(o.dates.to_numpy()[m], o.y[m], p[m], {}, o.tpos[m])
        save_preds(path, pr)
        log.info("%s mstl_arima %s: %d origins", ctx.zone, split, len(pr.y))


def all_artifact_names(zone: str, cfg: Config) -> list[str]:
    root = cfg.path("artifacts") / "models"
    base = root / zone
    names = (
        sorted(p.name for p in base.iterdir() if (p / "manifest.json").exists())
        if base.exists()
        else []
    )
    if (root / "pooled" / "pooled_lstm" / "manifest.json").exists() and zone != "DE":
        names.append("pooled_lstm")
    return names


__all__ = ["run_backtest", "get_preds", "ALL_REGIONS", "artifact_dir"]

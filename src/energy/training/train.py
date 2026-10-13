"""Fit models for a zone, validate, and save versioned artifacts with manifests."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

from energy.config import Config, load_config, set_seed
from energy.data.build import processed_path
from energy.evaluation.backtest import Context, collect, make_context, score
from energy.models.base import Forecaster
from energy.models.gbm import GBMForecaster
from energy.models.lstm import LSTMForecaster
from energy.models.naive import SeasonalNaive
from energy.training.manifest import write_manifest

log = logging.getLogger(__name__)

KINDS = ("naive_week", "naive_day", "lightgbm", "lstm")


def artifact_dir(cfg: Config, zone: str, name: str) -> Path:
    p = cfg.path("artifacts") / "models" / zone / name
    p.mkdir(parents=True, exist_ok=True)
    return p


def hpo_dir(cfg: Config) -> Path:
    p = cfg.path("results") / "hpo"
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_best_hp(kind: str, cfg: Config) -> dict[str, Any] | None:
    p = hpo_dir(cfg) / f"best_{kind}.json"
    if p.exists():
        d: dict[str, Any] = json.loads(p.read_text())
        int_keys = {
            "num_leaves",
            "min_child_samples",
            "n_estimators",
            "hidden",
            "layers",
            "batch_size",
        }
        return {k: int(v) if k in int_keys else v for k, v in d["params"].items()}
    return None


def fit_model(
    kind: str, ctx: Context, seed: int = 0, hp: dict[str, Any] | None = None, **opts: Any
) -> tuple[Forecaster, dict[str, Any]]:
    """Fit one model of the given kind on the training split with validation for tuning."""
    set_seed(seed)
    cfg = ctx.cfg
    if hp is None and kind in ("lstm", "lightgbm"):
        hp = load_best_hp(kind, cfg)
    offsets = tuple(opts.pop("phase_offsets", (0, 6, 12, 18)))
    train = ctx.origins("train", offsets=offsets if kind == "lstm" else (0,))
    val = ctx.origins("val")
    model: Forecaster
    t0 = time.time()
    if kind.startswith("naive"):
        model = SeasonalNaive(kind.split("_")[1])
    elif kind == "lightgbm":
        model = GBMForecaster(cfg, params=hp, seed=seed, **opts)
    elif kind == "lstm":
        model = LSTMForecaster(cfg, seed=seed, hp=hp, phase_offsets=offsets, **opts)
    else:
        raise ValueError(kind)
    model.fit(ctx.zf, train, val)
    elapsed = time.time() - t0
    pr = collect(model, ctx, val)
    metrics = score(pr, ctx.scale)
    metrics["fit_seconds"] = elapsed
    log.info(
        "%s %s val mae %.1f mape %.2f (%.0fs)",
        ctx.zone,
        kind,
        metrics["mae"],
        metrics["mape"],
        elapsed,
    )
    return model, metrics


def save_artifact(
    name: str, ctx: Context, model: Forecaster, val_metrics: dict[str, Any], seed: int = 0
) -> Path:
    folder = artifact_dir(ctx.cfg, ctx.zone, name)
    if isinstance(model, GBMForecaster | LSTMForecaster):
        model.save(folder)
        if isinstance(model, LSTMForecaster):
            import torch

            from energy.explain.shap_lstm import sample_background

            # SHAP background (100 training windows) travels with the model so that the
            # refresh job can explain forecasts without access to the training data.
            torch.save(sample_background(model, ctx, 100, 0), folder / "background.pt")
    else:
        resid = getattr(model, "_resid", None) or {}
        (folder / "model.json").write_text(
            json.dumps(
                {
                    "variant": getattr(model, "variant", ""),
                    "resid": {str(q): v.tolist() for q, v in resid.items()},
                }
            )
        )
    write_manifest(
        folder,
        model=name,
        zone=ctx.zone,
        cfg=ctx.cfg,
        data_path=processed_path(ctx.zone, ctx.cfg),
        seed=seed,
        train_period=(ctx.dates["train"][0], ctx.dates["train"][1]),
        val_metrics=val_metrics,
    )
    return folder


def load_artifact(kind: str, zone: str, cfg: Config, name: str | None = None) -> Forecaster:
    folder = artifact_dir(cfg, zone, name or kind)
    if kind.startswith("naive"):
        m = SeasonalNaive(kind.split("_")[1])
        meta = json.loads((folder / "model.json").read_text())
        if meta.get("resid"):
            m._resid = {float(q): np.array(v) for q, v in meta["resid"].items()}
        return m
    if kind == "lstm":
        return LSTMForecaster.load(folder, cfg)
    if kind == "lightgbm":
        import lightgbm as lgb

        meta = json.loads((folder / "meta.json").read_text())
        features = meta.get("features")
        if features is None and name:
            # artifacts written before the feature list was stored: use the registry
            from energy.training.ablations import ABLATIONS

            features = ABLATIONS.get(name, ("", {}))[1].get("features")
        g = GBMForecaster(cfg, params=meta["params"], seed=meta["seed"], features=features)
        g.best_iter = meta["best_iter"]
        for k in g.best_iter:
            g.models[k] = lgb.Booster(model_file=str(folder / f"{k}.txt"))
        return g
    raise ValueError(kind)


def train_zone(
    zone: str,
    kinds: tuple[str, ...] = ("naive_week", "naive_day", "lightgbm", "lstm"),
    seed: int = 0,
    cfg: Config | None = None,
) -> dict[str, dict[str, Any]]:
    cfg = cfg or load_config()
    ctx = make_context(zone, cfg)
    out = {}
    for k in kinds:
        model, m = fit_model(k, ctx, seed=seed)
        save_artifact(k, ctx, model, m, seed)
        out[k] = m
    return out


def train_many(zones: list[str], kinds: list[str], seeds: list[int]) -> None:
    """Resumable: skips artifacts whose manifest already exists."""
    cfg = load_config()
    for z in zones:
        ctx = make_context(z, cfg)
        for k in kinds:
            for s in seeds if k == "lstm" else [0]:
                name = k if s == 0 else f"{k}_s{s}"
                if (artifact_dir(cfg, z, name) / "manifest.json").exists():
                    log.info("skip %s %s (exists)", z, name)
                    continue
                model, m = fit_model(k, ctx, seed=s)
                save_artifact(name, ctx, model, m, s)

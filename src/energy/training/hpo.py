"""Small random hyperparameter searches, selected on the validation split only."""

from __future__ import annotations

import json
import logging
import time
from typing import Any

import numpy as np
import pandas as pd

from energy.config import Config, load_config
from energy.evaluation.backtest import collect, make_context, score
from energy.training.train import fit_model, hpo_dir, load_best_hp

log = logging.getLogger(__name__)


def space_lstm(rng: np.random.Generator) -> dict[str, Any]:
    return {
        "hidden": int(rng.choice([64, 96, 128])),
        "layers": int(rng.choice([1, 2])),
        "dropout": float(rng.choice([0.0, 0.1, 0.2])),
        "lr": float(np.exp(rng.uniform(np.log(5e-4), np.log(4e-3)))),
        "weight_decay": float(rng.choice([1e-5, 1e-4, 1e-3])),
        "batch_size": int(rng.choice([128, 256])),
    }


def space_gbm(rng: np.random.Generator) -> dict[str, Any]:
    return {
        "num_leaves": int(rng.choice([15, 31, 63, 127])),
        "learning_rate": float(rng.choice([0.02, 0.05, 0.1])),
        "min_child_samples": int(rng.choice([20, 40, 100])),
        "subsample": float(rng.choice([0.6, 0.8, 1.0])),
        "colsample_bytree": float(rng.choice([0.6, 0.8, 1.0])),
        "n_estimators": 1500,
    }


def run_hpo(
    kind: str,
    zone: str = "DE",
    n_trials: int = 20,
    max_epochs: int = 10,
    cfg: Config | None = None,
) -> pd.DataFrame:
    """Random search; resumable (completed trials are read back from the CSV)."""
    cfg = cfg or load_config()
    ctx = make_context(zone, cfg)
    path = hpo_dir(cfg) / f"{kind}_{zone}.csv"
    done = pd.read_csv(path) if path.exists() else pd.DataFrame()
    rng = np.random.default_rng(cfg.seed)
    space = space_lstm if kind == "lstm" else space_gbm
    rows = done.to_dict("records")
    for i in range(n_trials):
        hp = space(rng)  # always draw so the sequence is reproducible when resuming
        if i < len(done):
            continue
        t0 = time.time()
        hp_run = {**hp, "max_epochs": max_epochs, "patience": 3} if kind == "lstm" else hp
        # Reduced protocol for the LSTM search: two issue phases instead of four.
        extra = {"phase_offsets": (0, 12)} if kind == "lstm" else {}
        model, met = fit_model(kind, ctx, seed=cfg.seed, hp=hp_run, **extra)
        if kind == "lstm":
            sel = float(model.best_val_loss)  # type: ignore[attr-defined]
        else:
            sel = met["mae"]
        row = {
            "trial": i,
            **hp,
            "val_mae": met["mae"],
            "val_mape": met["mape"],
            "select": sel,
            "seconds": time.time() - t0,
        }
        rows.append(row)
        pd.DataFrame(rows).to_csv(path, index=False)
        log.info("hpo %s trial %d select %.4f mae %.1f", kind, i, sel, met["mae"])
    df = pd.DataFrame(rows)
    best = df.loc[df["select"].idxmin()]
    keys = list(space(np.random.default_rng(0)).keys())
    params = {k: (best[k].item() if hasattr(best[k], "item") else best[k]) for k in keys}
    (hpo_dir(cfg) / f"best_{kind}.json").write_text(
        json.dumps(
            {
                "zone": zone,
                "criterion": "val pinball (lstm) / val MAE (lightgbm)",
                "trial": int(best["trial"]),
                "params": params,
            },
            indent=2,
        )
    )
    return df


__all__ = ["run_hpo", "load_best_hp", "collect", "score"]

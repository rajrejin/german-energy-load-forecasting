"""LightGBM direct day-ahead model: one row per (origin, target hour)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from energy.config import Config
from energy.features import GBM_FEATURES, Origins, ZoneFrame, to_long_frame
from energy.models.base import Forecaster

LOAD_COLS = [
    "lag24",
    "lag48",
    "lag72",
    "lag168",
    "lag336",
    "latest_same_hour",
    "last_obs_load",
    "load_mean24",
    "load_prev24_mean",
    "load_max24",
    "load_min24",
    "load_std168",
]
REF = "load_mean168"


def prepare(
    o: Origins, features: list[str] | None = None, drop_prefix: tuple[str, ...] = ()
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """Return (X, y_relative, ref_level, valid_row). Load features and the target are expressed
    relative to the trailing weekly mean so that level shifts (trend, crisis year) do not have
    to be extrapolated by the trees. The reference level itself stays a feature."""
    df = to_long_frame(o)
    ref = df[REF].to_numpy()
    for c in LOAD_COLS:
        df[c] = df[c] / df[REF]
    feats = list(features or GBM_FEATURES)
    if REF not in feats:
        feats.append(REF)
    feats = [f for f in feats if not f.startswith(drop_prefix)] if drop_prefix else feats
    X = df[feats].astype("float32")
    y = (df["y"].to_numpy() / ref).astype("float32")
    return X, y, ref, df["valid"].to_numpy()


class GBMForecaster(Forecaster):
    name = "lightgbm"

    def __init__(
        self,
        cfg: Config,
        params: dict[str, Any] | None = None,
        features: list[str] | None = None,
        seed: int = 0,
    ):
        self.cfg = cfg
        self.params = {**cfg.gbm.model_dump(), **(params or {})}
        self.features = features
        self.seed = seed
        self.models: dict[str, lgb.Booster] = {}
        self.best_iter: dict[str, int] = {}

    def _train_one(
        self, objective: str, alpha: float | None, Xt, yt, Xv, yv
    ) -> tuple[lgb.Booster, int]:
        p = self.params
        params: dict[str, Any] = {
            "objective": objective,
            "learning_rate": p["learning_rate"],
            "num_leaves": p["num_leaves"],
            "min_child_samples": p["min_child_samples"],
            "bagging_fraction": p["subsample"],
            "bagging_freq": 1,
            "feature_fraction": p["colsample_bytree"],
            "verbosity": -1,
            "seed": self.seed,
            "num_threads": 2,
            "deterministic": True,
            "force_row_wise": True,
        }
        if alpha is not None:
            params["alpha"] = alpha
        dt = lgb.Dataset(Xt, yt)
        callbacks: list[Any] = []
        valid_sets = []
        if Xv is not None:
            valid_sets = [lgb.Dataset(Xv, yv, reference=dt)]
            callbacks = [lgb.early_stopping(50, verbose=False)]
        b = lgb.train(
            params,
            dt,
            num_boost_round=p["n_estimators"],
            valid_sets=valid_sets,
            callbacks=callbacks,
        )
        return b, (b.best_iteration or p["n_estimators"])

    def fit(self, zf: ZoneFrame, train: Origins, val: Origins | None = None) -> GBMForecaster:
        Xt, yt, _, vt = prepare(train, self.features)
        Xt, yt = Xt[vt], yt[vt]
        Xv = yv = None
        if val is not None:
            Xv_all, yv_all, _, vv = prepare(val, self.features)
            Xv, yv = Xv_all[vv], yv_all[vv]
        for name, obj, alpha in (
            ("point", "regression", None),
            ("q10", "quantile", 0.1),
            ("q90", "quantile", 0.9),
        ):
            b, it = self._train_one(obj, alpha, Xt, yt, Xv, yv)
            self.models[name] = b
            self.best_iter[name] = it
        return self

    def _pred(self, name: str, o: Origins) -> np.ndarray:
        X, _, ref, _ = prepare(o, self.features)
        b = self.models[name]
        p = b.predict(X, num_iteration=self.best_iter[name]) * ref
        return np.asarray(p).reshape(len(o), 24)

    def predict(self, zf: ZoneFrame, o: Origins) -> np.ndarray:
        return self._pred("point", o)

    def predict_quantiles(
        self, zf: ZoneFrame, o: Origins, quantiles: tuple[float, ...] = (0.1, 0.5, 0.9)
    ) -> dict[float, np.ndarray]:
        q10, q90, med = self._pred("q10", o), self._pred("q90", o), self._pred("point", o)
        lo = np.minimum(q10, med)
        hi = np.maximum(q90, med)
        return {0.1: lo, 0.5: med, 0.9: hi}

    def feature_importance(self) -> dict[str, float]:
        b = self.models["point"]
        imp = b.feature_importance(importance_type="gain")
        return dict(zip(b.feature_name(), map(float, imp), strict=True))

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        for k, b in self.models.items():
            b.save_model(str(path / f"{k}.txt"))
        (path / "meta.json").write_text(
            json.dumps({"best_iter": self.best_iter, "params": self.params, "seed": self.seed})
        )

"""Rolling-origin backtest: one forecast per day, 24 hourly predictions each."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from energy.config import Config, load_config
from energy.data.build import load_table
from energy.evaluation import metrics as M
from energy.features import Origins, ZoneFrame, build_origins
from energy.models.base import Forecaster
from energy.splits import split_dates

log = logging.getLogger(__name__)


@dataclass
class Context:
    zone: str
    cfg: Config
    zf: ZoneFrame
    last_date: pd.Timestamp
    dates: dict[str, tuple[str, str]]
    scale: float  # MASE scale: in-sample seasonal naive (168 h) MAE on the training period

    def origins(
        self,
        split: str,
        regime: str = "reanalysis",
        offsets: tuple[int, ...] = (0,),
        lookback: int | None = None,
    ) -> Origins:
        s, e = self.dates[split]
        return build_origins(self.zf, self.cfg, s, e, regime, offsets, lookback)


def make_context(zone: str, cfg: Config | None = None, df: pd.DataFrame | None = None) -> Context:
    cfg = cfg or load_config()
    if df is None:
        df = load_table(zone, cfg)
    zf = ZoneFrame(zone, df, cfg)
    last_valid = df["load"].last_valid_index()
    assert last_valid is not None
    last_date = last_valid.tz_convert("Europe/Berlin").normalize().tz_localize(None) - pd.Timedelta(
        days=1
    )
    dates = split_dates(cfg, last_date)
    train_end = pd.Timestamp(cfg.splits.train_end, tz="UTC") + pd.Timedelta(days=1)
    scale = M.mase_scale(df["load"].loc[:train_end].to_numpy())
    return Context(zone, cfg, zf, last_date, dates, scale)


@dataclass
class Preds:
    dates: np.ndarray  # (n,) datetime64 of target day
    y: np.ndarray  # (n, 24)
    p: np.ndarray  # (n, 24)
    q: dict[float, np.ndarray] = field(default_factory=dict)
    tpos: np.ndarray | None = None


def collect(model: Forecaster, ctx: Context, o: Origins, probabilistic: bool = True) -> Preds:
    m = o.valid
    p = model.predict(ctx.zf, o)[m]
    q: dict[float, np.ndarray] = {}
    if probabilistic:
        try:
            q = {k: v[m] for k, v in model.predict_quantiles(ctx.zf, o).items()}
        except NotImplementedError:
            q = {}
    return Preds(o.dates.to_numpy()[m], o.y[m], p, q, o.tpos[m])


def pred_path(cfg: Config, regime: str, zone: str, split: str, model: str) -> Path:
    p = cfg.path("processed") / "preds" / regime / zone / split / f"{model}.npz"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def save_preds(path: Path, pr: Preds) -> None:
    arrs: dict[str, Any] = {
        "dates": pr.dates.astype("datetime64[ns]").astype("int64"),
        "y": pr.y,
        "p": pr.p,
    }
    if pr.tpos is not None:
        arrs["tpos"] = pr.tpos
    for k, v in pr.q.items():
        arrs[f"q{int(round(k * 100)):02d}"] = v
    np.savez_compressed(path, **arrs)


def load_preds(path: Path) -> Preds:
    z = np.load(path)
    q = {int(k[1:]) / 100.0: z[k] for k in z.files if k.startswith("q")}
    return Preds(
        z["dates"].astype("datetime64[ns]"),
        z["y"],
        z["p"],
        q,
        z["tpos"] if "tpos" in z.files else None,
    )


def score(pr: Preds, scale: float) -> dict[str, float]:
    out = M.point_metrics(pr.y, pr.p, scale)
    out["n_origins"] = float(len(pr.y))
    if pr.q:
        out.update(M.quantile_metrics(pr.y, pr.q))
    return out

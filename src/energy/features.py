"""Feature construction for forecast origins, with explicit provenance for leakage tests.

Everything derived from observed load or observed weather only reads positions at or before
``last_obs_pos``. Target-day weather comes from the chosen regime ('reanalysis' proxy or
'archived' NWP forecast) and calendar features are deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from energy.config import Config
from energy.data.build import FC_PREFIX, WEATHER_VARS
from energy.windows import HORIZON, block_valid, nan_cumsum, schedule

REGIMES = ("reanalysis", "archived")
LOAD_LAGS = (48, 72, 168, 336)


class ZoneFrame:
    """Numpy view of one zone's hourly table."""

    def __init__(self, zone: str, df: pd.DataFrame, cfg: Config):
        self.zone = zone
        self.cfg = cfg
        self.df = df
        self.index = df.index
        self.T = len(df)
        self.load = df["load"].to_numpy(dtype=float)
        self.obs = df[WEATHER_VARS].to_numpy(dtype=float)
        self.fc = df[[FC_PREFIX + v for v in WEATHER_VARS]].to_numpy(dtype=float)
        self.hour = df["hour"].to_numpy()
        self.weekday = df["weekday"].to_numpy()
        self.doy = df["dayofyear"].to_numpy()
        self.hol = df["holiday_share"].to_numpy(dtype=float)
        self.bridge = df["bridge_day"].to_numpy(dtype=float)
        self.xmas = df["christmas_period"].to_numpy(dtype=float)
        self.is_dst = df["is_dst_transition"].to_numpy(dtype=bool)
        self.gen = {c: df[c].to_numpy(dtype=float) for c in df.columns if c.startswith("gen_")}
        self.load_nan_cum = nan_cumsum(self.load)
        self.obs_nan_cum = nan_cumsum(self.obs[:, 0])
        self.fc_nan_cum = nan_cumsum(self.fc[:, 0])

    def target_weather(self, regime: str) -> np.ndarray:
        if regime == "reanalysis":
            return self.obs
        if regime == "archived":
            return self.fc
        raise ValueError(f"unknown weather regime {regime!r}")


def hdd_cdd(temp: np.ndarray, cfg: Config) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.maximum(cfg.features.hdd_base_c - temp, 0.0),
        np.maximum(temp - cfg.features.cdd_base_c, 0.0),
    )


@dataclass
class Origins:
    zone: str
    regime: str
    sched: pd.DataFrame
    tpos: np.ndarray  # (n, 24) target positions
    y: np.ndarray  # (n, 24)
    valid: np.ndarray  # (n,)
    fut: dict[str, np.ndarray]  # name -> (n, 24)
    ctx: dict[str, np.ndarray]  # name -> (n,)
    provenance: dict[str, np.ndarray] = field(default_factory=dict)  # name -> max source pos
    lookback: int = 168

    def __len__(self) -> int:
        return len(self.sched)

    @property
    def dates(self) -> pd.DatetimeIndex:
        return pd.DatetimeIndex(self.sched["date"])


def build_origins(
    zf: ZoneFrame,
    cfg: Config,
    start_date: str,
    end_date: str,
    regime: str = "reanalysis",
    offsets: tuple[int, ...] = (0,),
    lookback: int | None = None,
    require_target: bool = True,
) -> Origins:
    L = lookback or cfg.lstm.lookback
    sch = schedule(zf.index, cfg, start_date, end_date, offsets)
    last = sch["last_obs_pos"].to_numpy(dtype=np.int64)
    ts = sch["tstart_pos"].to_numpy(dtype=np.int64)
    tpos = ts[:, None] + np.arange(HORIZON)[None, :]
    tpos = np.minimum(tpos, zf.T - 1)

    y = zf.load[tpos]
    tw = zf.target_weather(regime)[tpos]  # (n,24,8)

    fut: dict[str, np.ndarray] = {}
    prov: dict[str, np.ndarray] = {}
    for j, v in enumerate(WEATHER_VARS):
        fut["w_" + v] = tw[:, :, j]
    hdd, cdd = hdd_cdd(tw[:, :, 0], cfg)
    fut["hdd"], fut["cdd"] = hdd, cdd
    fut["hour"] = zf.hour[tpos].astype(float)
    fut["weekday"] = zf.weekday[tpos].astype(float)
    fut["doy"] = zf.doy[tpos].astype(float)
    fut["holiday_share"] = zf.hol[tpos]
    fut["bridge_day"] = zf.bridge[tpos]
    fut["christmas_period"] = zf.xmas[tpos]
    fut["is_dst_transition"] = zf.is_dst[tpos].astype(float)
    fut["horizon"] = np.broadcast_to(np.arange(HORIZON, dtype=float), tpos.shape).copy()
    fut["temp_target_day_mean"] = np.repeat(np.nanmean(tw[:, :, 0], axis=1, keepdims=True), 24, 1)

    # Load lags relative to the target hour, only where the source position is observed.
    for lag in LOAD_LAGS:
        src = tpos - lag
        avail = (src <= last[:, None]) & (src >= 0)
        vals = np.where(avail, zf.load[np.clip(src, 0, zf.T - 1)], np.nan)
        fut[f"lag{lag}"] = vals
        prov[f"lag{lag}"] = np.where(avail, src, -1)
    src24 = tpos - 24
    avail24 = (src24 <= last[:, None]) & (src24 >= 0)
    fut["lag24"] = np.where(avail24, zf.load[np.clip(src24, 0, zf.T - 1)], np.nan)
    prov["lag24"] = np.where(avail24, src24, -1)
    # latest same-clock-hour observation: t - 24k for the smallest k with t - 24k <= last_obs
    k = np.maximum(np.ceil((tpos - last[:, None]) / 24.0).astype(np.int64), 1)
    src_l = tpos - 24 * k
    fut["latest_same_hour"] = zf.load[np.clip(src_l, 0, zf.T - 1)]
    prov["latest_same_hour"] = src_l

    # Lagged generation (>= 48 h before the target hour, hence always observed at issue time).
    for gname, garr in zf.gen.items():
        for lag in (48, 168):
            src = tpos - lag
            fut[f"{gname}_lag{lag}"] = garr[np.clip(src, 0, zf.T - 1)]
            prov[f"{gname}_lag{lag}"] = src

    ctx: dict[str, np.ndarray] = {}
    idx = last[:, None] - np.arange(168)[None, :]
    idx = np.clip(idx, 0, zf.T - 1)
    w168 = zf.load[idx]
    ctx["last_obs_load"] = zf.load[last]
    ctx["load_mean24"] = w168[:, :24].mean(axis=1)
    ctx["load_prev24_mean"] = w168[:, 24:48].mean(axis=1)
    ctx["load_mean168"] = w168.mean(axis=1)
    ctx["load_std168"] = w168.std(axis=1)
    ctx["load_max24"] = w168[:, :24].max(axis=1)
    ctx["load_min24"] = w168[:, :24].min(axis=1)
    ctx["temp_obs_mean24"] = zf.obs[idx[:, :24], 0].mean(axis=1)
    ctx["temp_obs_last"] = zf.obs[last, 0]
    for name in ctx:
        prov[name] = last.copy()

    # Validity: complete history window, complete target load (if required), weather present.
    # The history window used for validity is fixed at 168 h (and 336 h must exist) so that
    # every lookback variant is evaluated on the same origins.
    ok = last - 335 >= 0
    lo168 = np.clip(last - 167, 0, None)
    t_end = np.minimum(ts + HORIZON - 1, zf.T - 1)
    ok &= block_valid(zf.load_nan_cum, lo168, last)
    ok &= block_valid(zf.obs_nan_cum, lo168, last)
    if regime == "reanalysis":
        ok &= block_valid(zf.obs_nan_cum, ts, t_end)
    else:
        ok &= block_valid(zf.fc_nan_cum, ts, t_end)
    if require_target:
        ok &= block_valid(zf.load_nan_cum, ts, np.minimum(ts + HORIZON - 1, zf.T - 1))
    ok &= ts + HORIZON - 1 < zf.T
    ok &= ~np.isnan(fut["lag168"]).any(axis=1)
    return Origins(zf.zone, regime, sch, tpos, y, ok, fut, ctx, prov, lookback=L)


def to_long_frame(o: Origins) -> pd.DataFrame:
    """Flatten origins to one row per (origin, horizon) for tabular models."""
    n = len(o)
    rows: dict[str, Any] = {
        "date": np.repeat(o.sched["date"].to_numpy(), HORIZON),
        "h": np.tile(np.arange(HORIZON), n),
        "tpos": o.tpos.reshape(-1),
        "y": o.y.reshape(-1),
    }
    for name, arr in o.fut.items():
        rows[name] = arr.reshape(-1)
    for name, arr in o.ctx.items():
        rows[name] = np.repeat(arr, HORIZON)
    df = pd.DataFrame(rows)
    df["origin"] = np.repeat(np.arange(n), HORIZON)
    df["valid"] = np.repeat(o.valid, HORIZON)
    ang = 2 * np.pi * df["doy"] / 365.25
    df["doy_sin"], df["doy_cos"] = np.sin(ang), np.cos(ang)
    ang_h = 2 * np.pi * df["hour"] / 24.0
    df["hour_sin"], df["hour_cos"] = np.sin(ang_h), np.cos(ang_h)
    df["is_weekend"] = (df["weekday"] >= 5).astype(float)
    return df


GBM_FEATURES = [
    *[f"w_{v}" for v in WEATHER_VARS],
    "hdd",
    "cdd",
    "hour",
    "weekday",
    "doy_sin",
    "doy_cos",
    "hour_sin",
    "hour_cos",
    "is_weekend",
    "holiday_share",
    "bridge_day",
    "christmas_period",
    "horizon",
    "temp_target_day_mean",
    "lag24",
    "lag48",
    "lag72",
    "lag168",
    "lag336",
    "latest_same_hour",
    "last_obs_load",
    "load_mean24",
    "load_prev24_mean",
    "load_mean168",
    "load_std168",
    "load_max24",
    "load_min24",
    "temp_obs_mean24",
    "temp_obs_last",
]

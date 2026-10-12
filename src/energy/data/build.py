"""Build tidy hourly tables per zone: load, weather, archived forecasts, calendar."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from energy.calendar_features import calendar_frame
from energy.config import ALL_REGIONS, ZONES, Config, load_config
from energy.data import openmeteo
from energy.data.smard import FORECAST_LOAD_FILTER, LOAD_FILTER, load_cached_series, load_hourly

log = logging.getLogger(__name__)

WEATHER_VARS = [
    "temperature_2m",
    "apparent_temperature",
    "relative_humidity_2m",
    "cloud_cover",
    "shortwave_radiation",
    "wind_speed_10m",
    "wind_speed_100m",
    "precipitation",
]
FC_PREFIX = "fc_"
# Generation groups (SMARD filters) used only for the lagged-generation ablation.
GEN_GROUPS: dict[str, list[int]] = {
    "gen_wind": [1225, 4067],
    "gen_pv": [4068],
    "gen_fossil": [1223, 4069, 4071, 1227],
}


def processed_path(zone: str, cfg: Config) -> Path:
    return cfg.path("processed") / f"{zone}.parquet"


def fill_short_gaps(s: pd.Series, max_gap: int) -> tuple[pd.Series, pd.Series]:
    """Linearly interpolate NaN runs of at most `max_gap` hours; returns (filled, flag)."""
    isn = s.isna()
    run_id = (isn != isn.shift()).cumsum()
    run_len = isn.groupby(run_id).transform("sum")
    short = isn & (run_len <= max_gap)
    # only interpolate interior gaps (need both neighbours)
    filled = s.interpolate(limit_area="inside", limit=max_gap)
    flag = short & filled.notna()
    out = s.copy()
    out[flag] = filled[flag]
    return out, flag


def _weighted_national(frames: dict[str, pd.DataFrame], weights: dict[str, float]) -> pd.DataFrame:
    tot = sum(weights.values())
    num = None
    den = None
    for z, df in frames.items():
        w = weights[z] / tot
        valid = df.notna().astype(float) * w
        part = df.fillna(0.0) * w
        num = part if num is None else num.add(part, fill_value=0.0)
        den = valid if den is None else den.add(valid, fill_value=0.0)
    assert num is not None and den is not None
    return num / den.where(den > 0)


def build_all(cfg: Config | None = None) -> dict[str, Path]:
    cfg = cfg or load_config()
    zw_obs: dict[str, pd.DataFrame] = {}
    zw_fc: dict[str, pd.DataFrame] = {}
    for z in ZONES:
        zw_obs[z] = openmeteo.zone_weather(openmeteo.ARCHIVE, z, cfg)
        zw_fc[z] = openmeteo.zone_weather(openmeteo.HISTFC, z, cfg)

    loads: dict[str, tuple[pd.Series, str]] = {
        r: load_hourly(LOAD_FILTER, r, cfg) for r in ALL_REGIONS
    }
    train_end = pd.Timestamp(cfg.splits.train_end, tz="UTC") + pd.Timedelta(days=1)
    weights = {z: float(loads[z][0].loc[:train_end].mean()) for z in ZONES}
    weights = {z: v for z, v in weights.items() if np.isfinite(v)}
    (cfg.path("results") / "national_weights.json").write_text(json.dumps(weights, indent=2))
    zw_obs["DE"] = _weighted_national({z: zw_obs[z] for z in weights}, weights)
    zw_fc["DE"] = _weighted_national({z: zw_fc[z] for z in weights}, weights)

    out: dict[str, Path] = {}
    for region in ALL_REGIONS:
        load, src = loads[region]
        if load.empty:
            log.warning("no load for %s", region)
            continue
        idx = pd.date_range(
            pd.Timestamp(cfg.data_start, tz="UTC"), load.index.max(), freq="1h", tz="UTC"
        )
        df = pd.DataFrame(index=idx)
        raw_load = load.reindex(idx)
        df["load_raw_missing"] = raw_load.isna()
        filled, flag = fill_short_gaps(raw_load, cfg.quality.max_interp_gap_h)
        df["load"] = filled
        df["load_interpolated"] = flag
        fc = load_cached_forecast(region, cfg)
        df["smard_load_forecast"] = fc.reindex(idx)
        w_obs = zw_obs[region].reindex(idx)
        w_fc = zw_fc[region].reindex(idx)
        for v in WEATHER_VARS:
            col = w_obs[v] if v in w_obs else pd.Series(np.nan, index=idx)
            df[v], df[v + "_interp"] = fill_short_gaps(col, cfg.quality.max_interp_gap_h)
            fcol = w_fc[v] if v in w_fc else pd.Series(np.nan, index=idx)
            df[FC_PREFIX + v] = fcol
        for col, filters in GEN_GROUPS.items():
            parts = [load_cached_series(f, region, "hour", cfg) for f in filters]
            parts = [s for s in parts if len(s)]
            if parts:
                tot = sum(s.reindex(idx) for s in parts)
                df[col] = tot  # NaN when any component is missing
        cal = calendar_frame(idx, region)
        df = df.join(cal)
        df["source"] = src
        df["zone"] = region
        p = processed_path(region, cfg)
        df.to_parquet(p)
        out[region] = p
        log.info("built %s: %d rows, load NaN %d", region, len(df), int(df["load"].isna().sum()))
    return out


def load_cached_forecast(region: str, cfg: Config) -> pd.Series:
    from energy.data.smard import load_cached_series

    return load_cached_series(FORECAST_LOAD_FILTER, region, "hour", cfg)


def load_table(zone: str, cfg: Config | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    p = processed_path(zone, cfg)
    if not p.exists():
        raise FileNotFoundError(f"{p} missing: run `energy build-dataset` after `energy fetch`")
    return pd.read_parquet(p)

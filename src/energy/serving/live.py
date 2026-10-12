"""Assemble a short live feature table per region from fresh SMARD actuals and Open-Meteo."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from energy.calendar_features import calendar_frame
from energy.config import ALL_REGIONS, ZONES, Config, load_zones
from energy.data import openmeteo
from energy.data.build import FC_PREFIX, WEATHER_VARS, _weighted_national, fill_short_gaps
from energy.data.smard import LOAD_FILTER, SmardClient, load_cached_series
from energy.features import ZoneFrame

log = logging.getLogger(__name__)

TZ = "Europe/Berlin"


def fetch_live_load(cfg: Config, days: int = 30, client: SmardClient | None = None) -> None:
    """Refresh the most recent SMARD chunks for the load of every region."""
    cli = client or SmardClient(cfg)
    start = (pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=days)).strftime("%Y-%m-%d")
    for r in ALL_REGIONS:
        cli.fetch_series(LOAD_FILTER, r, "hour", start=start)


def fetch_live_weather(
    cfg: Config, past_days: int = 14, forecast_days: int = 3, client: Any = None
) -> dict[str, pd.DataFrame]:
    """Population-weighted live weather per zone (past days and forecast), plus DE."""
    cli = client or openmeteo.OpenMeteoClient(cfg)
    zones_cfg = load_zones()["zones"]
    frames: dict[str, pd.DataFrame] = {}
    for z in ZONES:
        pts = {}
        for p in zones_cfg[z]["points"]:
            payload = cli.fetch_live(p["lat"], p["lon"], past_days, forecast_days)
            pts[p["name"]] = openmeteo.parse_hourly(payload)
        frames[z] = openmeteo.zone_weather("live", z, cfg, frames=pts)
    return frames


def national_weights(cfg: Config) -> dict[str, float]:
    """Mean training-period zone loads used to combine zone weather into national weather."""
    import json

    p = cfg.path("results") / "national_weights.json"
    if p.exists():
        d: dict[str, float] = json.loads(p.read_text())
        return d
    raise FileNotFoundError("results/national_weights.json missing: run `energy build-dataset`")


def build_live_frame(
    region: str,
    cfg: Config,
    weather: dict[str, pd.DataFrame],
    now: pd.Timestamp,
    back_hours: int = 24 * 20,
    fwd_hours: int = 24 * 3,
) -> ZoneFrame:
    end = now.floor("1h") + pd.Timedelta(hours=fwd_hours)
    idx = pd.date_range(now.floor("1h") - pd.Timedelta(hours=back_hours), end, freq="1h", tz="UTC")
    load = load_cached_series(LOAD_FILTER, region, "hour", cfg).reindex(idx)
    filled, _ = fill_short_gaps(load, cfg.quality.max_interp_gap_h)
    df = pd.DataFrame(index=idx)
    df["load"] = filled
    w = (
        weather[region]
        if region in weather
        else _weighted_national({z: weather[z] for z in ZONES}, national_weights(cfg))
    )
    w = w.reindex(idx)
    for v in WEATHER_VARS:
        # The live API returns model analysis for past hours and forecasts for future hours.
        df[v] = w[v]
        df[FC_PREFIX + v] = w[v]
    df = df.join(calendar_frame(idx, region))
    return ZoneFrame(region, df, cfg)


def issue_candidates(now: pd.Timestamp, cfg: Config, n_days: int = 4) -> list[pd.Timestamp]:
    """Issue instants (local 10:00 on recent days) that are not in the future, newest first."""
    loc_now = now.tz_convert(TZ)
    out = []
    for k in range(n_days):
        d = (loc_now.normalize() - pd.Timedelta(days=k)).tz_localize(None)
        issue = (d + pd.Timedelta(hours=cfg.issue_hour_local)).tz_localize(TZ).tz_convert("UTC")
        if issue <= now:
            out.append(issue)
    return out


def pick_issue(
    zf: ZoneFrame, now: pd.Timestamp, cfg: Config, regime: str = "archived"
) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    """Latest issue slot for which load is known through the hour before issue and the full
    target block has weather. Returns (issue_utc, target_date_local) or (None, None)."""
    from energy.features import build_origins

    for issue in issue_candidates(now, cfg):
        target = issue.tz_convert(TZ).normalize().tz_localize(None) + pd.Timedelta(days=1)
        d = target.strftime("%Y-%m-%d")
        o = build_origins(zf, cfg, d, d, regime, require_target=False)
        if len(o) and o.valid[0]:
            return issue, target
    return None, None


__all__ = ["fetch_live_load", "fetch_live_weather", "build_live_frame", "pick_issue", "np"]

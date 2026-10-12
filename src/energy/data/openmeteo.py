"""Open-Meteo clients (reanalysis archive, archived forecasts, live forecast) with a cache."""

from __future__ import annotations

import gzip
import json
import logging
import time
from pathlib import Path
from typing import Any

import httpx
import pandas as pd
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from energy.config import ZONES, Config, load_config, load_zones

log = logging.getLogger(__name__)

ARCHIVE = "archive"
HISTFC = "histforecast"


class OpenMeteoClient:
    def __init__(self, cfg: Config | None = None, client: httpx.Client | None = None):
        self.cfg = cfg or load_config()
        self.root = self.cfg.path("raw") / "openmeteo"
        self.http = client or httpx.Client(
            timeout=60.0, headers={"User-Agent": "energy-forecaster"}
        )

    def _base(self, kind: str) -> str:
        return {
            ARCHIVE: self.cfg.openmeteo.archive_url,
            HISTFC: self.cfg.openmeteo.hist_forecast_url,
            "live": self.cfg.openmeteo.forecast_url,
        }[kind]

    def cache_path(self, kind: str, point: str, year: int) -> Path:
        return self.root / kind / point / f"{year}.json.gz"

    @retry(
        retry=retry_if_exception_type((httpx.TransportError, httpx.HTTPStatusError, ValueError)),
        stop=stop_after_attempt(12),
        wait=wait_exponential(multiplier=5, min=5, max=300),
        reraise=True,
    )
    def _get(self, url: str, params: dict[str, Any]) -> dict[str, Any]:
        r = self.http.get(url, params=params)
        time.sleep(self.cfg.openmeteo.delay_s)
        r.raise_for_status()
        out: dict[str, Any] = r.json()
        return out

    def fetch_year(
        self, kind: str, point: str, lat: float, lon: float, year: int, force: bool = False
    ) -> Path | None:
        """Fetch one calendar year for one location. Current-year chunks are refetched."""
        path = self.cache_path(kind, point, year)
        today = pd.Timestamp.now(tz="UTC").normalize()
        if path.exists() and not force:
            with gzip.open(path, "rt", encoding="utf-8") as f:
                w = json.load(f)
            if w["fetched_at"] > pd.Timestamp(f"{year + 1}-01-15", tz="UTC").timestamp():
                return path
            if time.time() - w["fetched_at"] < 6 * 3600:
                return path
        start = pd.Timestamp(f"{year}-01-01", tz="UTC")
        end = min(pd.Timestamp(f"{year}-12-31", tz="UTC"), today)
        if kind == ARCHIVE:
            end = min(end, today - pd.Timedelta(days=2))
        if end < start:
            return None
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": start.strftime("%Y-%m-%d"),
            "end_date": end.strftime("%Y-%m-%d"),
            "hourly": ",".join(self.cfg.openmeteo.variables),
            "timezone": "UTC",
        }
        try:
            data = self._get(self._base(kind), params)
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 400:
                log.warning("open-meteo 400 %s %s %s: %s", kind, point, year, e.response.text[:200])
                return None
            raise
        path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(path, "wt", encoding="utf-8") as f:
            json.dump({"fetched_at": time.time(), "params": params, "payload": data}, f)
        return path

    def fetch_live(
        self, lat: float, lon: float, past_days: int = 3, forecast_days: int = 3
    ) -> dict:
        params = {
            "latitude": lat,
            "longitude": lon,
            "hourly": ",".join(self.cfg.openmeteo.variables),
            "timezone": "UTC",
            "past_days": past_days,
            "forecast_days": forecast_days,
        }
        return self._get(self._base("live"), params)


def parse_hourly(payload: dict[str, Any]) -> pd.DataFrame:
    h = payload["hourly"]
    idx = pd.to_datetime(h["time"], utc=True)
    df = pd.DataFrame({k: v for k, v in h.items() if k != "time"}, index=idx)
    return df.astype("float64")


def load_point(kind: str, point: str, cfg: Config | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    folder = cfg.path("raw") / "openmeteo" / kind / point
    parts = []
    for p in sorted(folder.glob("*.json.gz")):
        with gzip.open(p, "rt", encoding="utf-8") as f:
            w = json.load(f)
        parts.append(parse_hourly(w["payload"]))
    if not parts:
        return pd.DataFrame(index=pd.DatetimeIndex([], tz="UTC"))
    out = pd.concat(parts)
    out = out[~out.index.duplicated(keep="last")].sort_index()
    return out


def zone_weights(zone: str) -> dict[str, float]:
    pts = load_zones()["zones"][zone]["points"]
    tot = sum(p["pop"] for p in pts)
    return {p["name"]: p["pop"] / tot for p in pts}


def zone_weather(
    kind: str, zone: str, cfg: Config | None = None, frames: dict[str, pd.DataFrame] | None = None
) -> pd.DataFrame:
    """Population-weighted mean of each variable over the zone's points.

    A weighted mean over rows where a point is missing re-normalises weights over the
    points that are present.
    """
    w = zone_weights(zone)
    frames = frames or {name: load_point(kind, name, cfg) for name in w}
    cols = None
    num = None
    den = None
    for name, wt in w.items():
        df = frames[name]
        if df.empty:
            continue
        cols = df.columns if cols is None else cols
        valid = df.notna().astype(float) * wt
        part = df.fillna(0.0) * wt
        num = part if num is None else num.add(part, fill_value=0.0)
        den = valid if den is None else den.add(valid, fill_value=0.0)
    if num is None or den is None:
        return pd.DataFrame(index=pd.DatetimeIndex([], tz="UTC"))
    out = (num / den.where(den > 0)).sort_index()
    return out


def fetch_all(
    kinds: tuple[str, ...] = (ARCHIVE, HISTFC),
    start_year: int = 2015,
    hist_start_year: int = 2021,
    cfg: Config | None = None,
    zones: list[str] | None = None,
) -> None:
    cfg = cfg or load_config()
    cli = OpenMeteoClient(cfg)
    zones_cfg = load_zones()["zones"]
    this_year = pd.Timestamp.now(tz="UTC").year
    for kind in kinds:
        y0 = start_year if kind == ARCHIVE else hist_start_year
        for z in zones or ZONES:
            for p in zones_cfg[z]["points"]:
                for y in range(y0, this_year + 1):
                    cli.fetch_year(kind, p["name"], p["lat"], p["lon"], y)
                log.info("weather %s %s %s done", kind, z, p["name"])

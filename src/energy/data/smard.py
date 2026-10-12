"""SMARD (Bundesnetzagentur) chart_data client with a gzip JSON cache.

Endpoint documented by the community project bundesAPI/smard-api (unofficial); behaviour
was verified at run time (see docs/DECISIONS.md).
"""

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

from energy.config import Config, load_config

log = logging.getLogger(__name__)

FILTERS: dict[int, str] = {
    410: "consumption_total",
    411: "consumption_forecast",
    4359: "residual_load",
    4387: "pumped_storage_consumption",
    1223: "lignite",
    1224: "nuclear",
    1225: "wind_offshore",
    1226: "hydropower",
    1227: "other_conventional",
    1228: "other_renewables",
    4066: "biomass",
    4067: "wind_onshore",
    4068: "photovoltaics",
    4069: "hard_coal",
    4070: "pumped_storage",
    4071: "natural_gas",
}
LOAD_FILTER = 410
FORECAST_LOAD_FILTER = 411
GENERATION_FILTERS = [f for f in FILTERS if f not in (410, 411)]

RES_FREQ = {"hour": "1h", "quarterhour": "15min"}


class NotAvailableError(Exception):
    """The filter/region/resolution combination does not exist on the server."""


class SmardClient:
    def __init__(self, cfg: Config | None = None, client: httpx.Client | None = None):
        self.cfg = cfg or load_config()
        self.base = self.cfg.smard.base_url
        self.delay = self.cfg.smard.delay_s
        self.root = self.cfg.path("raw") / "smard"
        self.http = client or httpx.Client(
            timeout=30.0, headers={"User-Agent": "energy-forecaster"}
        )

    # -- paths -------------------------------------------------------------------------
    def cache_path(self, filt: int, region: str, res: str, ts: int | None) -> Path:
        name = "index" if ts is None else str(ts)
        return self.root / str(filt) / region / res / f"{name}.json.gz"

    # -- network -----------------------------------------------------------------------
    @retry(
        retry=retry_if_exception_type((httpx.TransportError, httpx.HTTPStatusError)),
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=1, max=30),
        reraise=True,
    )
    def _get_json(self, url: str) -> dict[str, Any]:
        r = self.http.get(url)
        time.sleep(self.delay)
        if r.status_code == 404:
            raise NotAvailableError(url)
        if r.status_code >= 500 or r.status_code == 429:
            r.raise_for_status()
        r.raise_for_status()
        out: dict[str, Any] = r.json()
        return out

    def _url(self, filt: int, region: str, res: str, ts: int | None) -> str:
        if ts is None:
            return f"{self.base}/{filt}/{region}/index_{res}.json"
        return f"{self.base}/{filt}/{region}/{filt}_{region}_{res}_{ts}.json"

    def fetch_index(self, filt: int, region: str, res: str = "hour") -> list[int]:
        data = self._get_json(self._url(filt, region, res, None))
        return [int(t) for t in data["timestamps"]]

    def fetch_chunk(
        self, filt: int, region: str, res: str, ts: int, force: bool = False
    ) -> Path | None:
        """Fetch one chunk into the cache unless a final copy already exists."""
        path = self.cache_path(filt, region, res, ts)
        now_ms = int(time.time() * 1000)
        if path.exists() and not force:
            with gzip.open(path, "rt", encoding="utf-8") as f:
                wrapper = json.load(f)
            # A chunk fetched before the revision window closed is refetched.
            window = self.cfg.smard.refetch_recent_days * 86_400_000
            if wrapper["fetched_at"] - ts > 7 * 86_400_000 + window:
                return path
            if now_ms - wrapper["fetched_at"] < 3_600_000:
                return path
        data = self._get_json(self._url(filt, region, res, ts))
        path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(path, "wt", encoding="utf-8") as f:
            json.dump({"fetched_at": now_ms, "payload": data}, f)
        return path

    def fetch_series(
        self,
        filt: int,
        region: str,
        res: str = "hour",
        start: str = "2015-01-01",
        end: pd.Timestamp | None = None,
    ) -> str:
        """Fetch every chunk between start and now. Returns 'ok', 'unavailable' or 'empty'."""
        try:
            idx = self.fetch_index(filt, region, res)
        except NotAvailableError:
            return "unavailable"
        now = pd.Timestamp.now(tz="UTC") if end is None else end
        lo = int((pd.Timestamp(start, tz="UTC") - pd.Timedelta(days=8)).timestamp() * 1000)
        hi = int(now.timestamp() * 1000)
        todo = [t for t in idx if lo <= t <= hi]
        n_ok = 0
        for t in todo:
            try:
                self.fetch_chunk(filt, region, res, t)
                n_ok += 1
            except NotAvailableError:
                log.warning("chunk missing %s %s %s %s", filt, region, res, t)
        log.info("smard %s %s %s: %d/%d chunks", filt, region, res, n_ok, len(todo))
        return "ok" if n_ok else "empty"


# -- parsing --------------------------------------------------------------------------------
def parse_chunk(payload: dict[str, Any]) -> pd.Series:
    """Parse one chunk payload ({'series': [[ts_ms, value|null], ...]}) into a UTC series."""
    rows = payload.get("series", [])
    if not rows:
        return pd.Series(dtype="float64", name="value")
    ts = pd.to_datetime([r[0] for r in rows], unit="ms", utc=True)
    vals = [float(r[1]) if r[1] is not None else float("nan") for r in rows]
    return pd.Series(vals, index=ts, name="value", dtype="float64")


def load_cached_series(
    filt: int, region: str, res: str = "hour", cfg: Config | None = None
) -> pd.Series:
    """Concatenate all cached chunks for one series. Later fetches win on duplicates."""
    cfg = cfg or load_config()
    folder = cfg.path("raw") / "smard" / str(filt) / region / res
    parts: list[pd.Series] = []
    for p in sorted(folder.glob("*.json.gz"), key=lambda x: x.stem):
        if p.name.startswith("index"):
            continue
        with gzip.open(p, "rt", encoding="utf-8") as f:
            wrapper = json.load(f)
        s = parse_chunk(wrapper["payload"])
        if len(s):
            parts.append(s)
    if not parts:
        return pd.Series(dtype="float64", name="value", index=pd.DatetimeIndex([], tz="UTC"))
    out = pd.concat(parts)
    out = out[~out.index.duplicated(keep="last")].sort_index().dropna()
    out.name = f"{filt}_{region}"
    return out


def quarterhour_to_hourly(s: pd.Series) -> pd.Series:
    """Aggregate a 15-minute power series to hourly mean power.

    Hours with fewer than four quarter-hour values are dropped (not extrapolated).
    """
    g = s.resample("1h")
    out: pd.Series = g.mean()
    cnt = g.count()
    return out[cnt == 4].dropna()


def load_hourly(filt: int, region: str, cfg: Config | None = None) -> tuple[pd.Series, str]:
    """Hourly series for a filter/region. Falls back to aggregated quarter-hours for gaps.

    Returns the series and the source label ('hour', or 'hour+quarterhour').
    """
    h = load_cached_series(filt, region, "hour", cfg)
    q = load_cached_series(filt, region, "quarterhour", cfg)
    if len(q) == 0:
        return h, "hour"
    qh = quarterhour_to_hourly(q)
    missing = qh.index.difference(h.index)
    if len(missing) == 0:
        return h, "hour"
    out = pd.concat([h, qh.loc[missing]]).sort_index()
    return out, "hour+quarterhour"

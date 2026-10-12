"""Orchestrates raw data acquisition and writes results/data_availability.csv."""

from __future__ import annotations

import logging

import pandas as pd

from energy.config import ALL_REGIONS, Config, load_config
from energy.data import openmeteo
from energy.data.smard import (
    FILTERS,
    FORECAST_LOAD_FILTER,
    GENERATION_FILTERS,
    LOAD_FILTER,
    SmardClient,
    load_cached_series,
)

log = logging.getLogger(__name__)


def fetch_smard(
    filters: list[int], regions: list[str], start: str, cfg: Config, resolution: str = "hour"
) -> list[dict]:
    cli = SmardClient(cfg)
    rows = []
    for f in filters:
        for r in regions:
            status = cli.fetch_series(f, r, resolution, start=start)
            rows.append({"filter": f, "name": FILTERS.get(f, "?"), "region": r, "status": status})
    return rows


def write_availability(cfg: Config) -> pd.DataFrame:
    rows = []
    for f in FILTERS:
        for r in ALL_REGIONS:
            s = load_cached_series(f, r, "hour", cfg)
            rows.append(
                {
                    "filter": f,
                    "name": FILTERS[f],
                    "region": r,
                    "n_hours": len(s),
                    "first": s.index[0].isoformat() if len(s) else "",
                    "last": s.index[-1].isoformat() if len(s) else "",
                }
            )
    df = pd.DataFrame(rows)
    out = cfg.path("results") / "data_availability.csv"
    df.to_csv(out, index=False)
    return df


def run_fetch(what: str = "all", slice_only: bool = False, cfg: Config | None = None) -> None:
    cfg = cfg or load_config()
    start = "2023-01-01" if slice_only else cfg.data_start
    if what in ("load", "all"):
        fetch_smard([LOAD_FILTER, FORECAST_LOAD_FILTER], ALL_REGIONS, start, cfg)
    if what in ("weather", "all"):
        if slice_only:
            from energy.config import load_zones

            p = load_zones()["zones"]["TenneT"]["points"][0]
            cli = openmeteo.OpenMeteoClient(cfg)
            for y in (2023, 2024, 2025, 2026):
                cli.fetch_year(openmeteo.ARCHIVE, p["name"], p["lat"], p["lon"], y)
        else:
            openmeteo.fetch_all(cfg=cfg)
    if what in ("generation", "all") and not slice_only:
        fetch_smard(GENERATION_FILTERS, ALL_REGIONS, start, cfg)
    write_availability(cfg)

"""Orchestrates raw data acquisition and writes results/data_availability.csv."""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pandas as pd
import psutil

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


def lock_dir(cfg: Config) -> Path:
    p = cfg.path("raw") / ".locks"
    p.mkdir(parents=True, exist_ok=True)
    return p


@contextmanager
def fetch_lock(cfg: Config) -> Iterator[Path]:
    """Marks a running fetch with a per-process lock file, removed when the fetch ends."""
    path = lock_dir(cfg) / f"fetch-{os.getpid()}.lock"
    path.write_text(str(os.getpid()))
    try:
        yield path
    finally:
        path.unlink(missing_ok=True)


def active_fetches(cfg: Config) -> list[int]:
    """PIDs of fetches that are still running; lock files of dead processes are removed."""
    alive = []
    for f in lock_dir(cfg).glob("fetch-*.lock"):
        try:
            pid = int(f.read_text().strip())
        except (OSError, ValueError):
            continue
        if psutil.pid_exists(pid):
            alive.append(pid)
        else:
            f.unlink(missing_ok=True)
    return sorted(alive)


def wait_for_fetches(cfg: Config | None = None, poll_s: float = 20.0) -> None:
    """Block until no fetch is running (used by the compute queue scripts)."""
    cfg = cfg or load_config()
    while active_fetches(cfg):
        time.sleep(poll_s)


def run_fetch(what: str = "all", slice_only: bool = False, cfg: Config | None = None) -> None:
    cfg = cfg or load_config()
    with fetch_lock(cfg):
        _run_fetch(what, slice_only, cfg)


def _run_fetch(what: str, slice_only: bool, cfg: Config) -> None:
    start = "2023-01-01" if slice_only else cfg.data_start
    if what in ("load", "all"):
        fetch_smard([LOAD_FILTER, FORECAST_LOAD_FILTER], ALL_REGIONS, start, cfg)
    if what in ("weather", "all"):
        if slice_only:
            from energy.config import load_zones

            p = load_zones()["zones"]["TenneT"]["points"][0]
            cli = openmeteo.OpenMeteoClient(cfg)
            for y in range(2023, pd.Timestamp.now(tz="UTC").year + 1):
                cli.fetch_year(openmeteo.ARCHIVE, p["name"], p["lat"], p["lon"], y)
        else:
            openmeteo.fetch_all(cfg=cfg)
    if what in ("generation", "all") and not slice_only:
        fetch_smard(GENERATION_FILTERS, ALL_REGIONS, start, cfg)
    write_availability(cfg)

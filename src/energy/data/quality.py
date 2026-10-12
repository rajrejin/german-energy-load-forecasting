"""Data quality checks over the processed tables; writes results/data_quality.json."""

from __future__ import annotations

import json
import logging
from typing import Any

import numpy as np
import pandas as pd

from energy.config import ALL_REGIONS, ZONES, Config, load_config
from energy.data.build import WEATHER_VARS, load_table

log = logging.getLogger(__name__)


def gap_runs(mask: pd.Series) -> list[tuple[pd.Timestamp, int]]:
    """Runs of True in a boolean series as (start, length)."""
    if not mask.any():
        return []
    rid = (mask != mask.shift()).cumsum()
    out = []
    for _, grp in mask[mask].groupby(rid[mask]):
        out.append((grp.index[0], len(grp)))
    return out


def robust_outliers(load: pd.Series, z_thresh: float) -> pd.Series:
    """Flag points whose residual, after removing the per-year weekday-hour median profile,
    has a robust z-score above the threshold."""
    df = pd.DataFrame({"y": load})
    loc = load.index.tz_convert("Europe/Berlin")
    df["key"] = list(zip(loc.year, loc.weekday, loc.hour, strict=True))
    prof = df.groupby("key")["y"].transform("median")
    r = df["y"] - prof
    med = r.median()
    mad = (r - med).abs().median()
    z = (r - med) / (1.4826 * mad) if mad > 0 else r * 0
    return z.abs() > z_thresh


def zone_report(zone: str, df: pd.DataFrame, cfg: Config) -> dict[str, Any]:
    load = df["load"]
    raw_missing = df["load_raw_missing"]
    after_missing = load.isna()
    long_gaps = gap_runs(after_missing)
    rep: dict[str, Any] = {
        "rows": int(len(df)),
        "first": df.index[0].isoformat(),
        "last": df.index[-1].isoformat(),
        "duplicate_timestamps": int(df.index.duplicated().sum()),
        "load_missing_raw_hours": int(raw_missing.sum()),
        "load_interpolated_hours": int(df["load_interpolated"].sum()),
        "load_missing_after_treatment_hours": int(after_missing.sum()),
        "long_gaps": [{"start": s.isoformat(), "hours": n} for s, n in long_gaps[:50]],
        "n_long_gaps": len(long_gaps),
        "load_min_mw": float(load.min()),
        "load_max_mw": float(load.max()),
        "load_mean_mw": float(load.mean()),
        "nonpositive_load_hours": int((load <= 0).sum()),
        "dst_transition_hours": int(df["is_dst_transition"].sum()),
        "weather_missing_hours": {
            v: int(df[v].isna().sum()) for v in WEATHER_VARS if v in df.columns
        },
        "weather_interpolated_hours": {
            v: int(df[v + "_interp"].sum()) for v in WEATHER_VARS if v + "_interp" in df.columns
        },
        "forecast_weather_first": (
            df["fc_temperature_2m"].first_valid_index().isoformat()
            if df["fc_temperature_2m"].notna().any()
            else None
        ),
        "smard_load_forecast_hours": int(df["smard_load_forecast"].notna().sum()),
    }
    out = robust_outliers(load.dropna(), cfg.quality.outlier_z)
    rep["outlier_hours"] = int(out.sum())
    rep["outlier_examples"] = [t.isoformat() for t in out[out].index[:10]]
    # yearly coverage
    cov = load.notna().groupby(load.index.year).mean()
    rep["coverage_by_year"] = {str(int(k)): round(float(v), 5) for k, v in cov.items()}
    return rep


def zone_sum_check(tables: dict[str, pd.DataFrame]) -> dict[str, Any]:
    if "DE" not in tables or not all(z in tables for z in ZONES):
        return {"available": False}
    s = sum(tables[z]["load"] for z in ZONES)
    de = tables["DE"]["load"]
    ok = s.notna() & de.notna()
    rel = ((s[ok] - de[ok]) / de[ok]).abs()
    signed = (s[ok] - de[ok]) / de[ok]
    return {
        "available": True,
        "hours_compared": int(ok.sum()),
        "mean_abs_rel_diff": float(rel.mean()),
        "p99_abs_rel_diff": float(rel.quantile(0.99)),
        "max_abs_rel_diff": float(rel.max()),
        "mean_signed_rel_diff": float(signed.mean()),
    }


def run_quality(cfg: Config | None = None) -> dict[str, Any]:
    cfg = cfg or load_config()
    tables = {}
    for r in ALL_REGIONS:
        try:
            tables[r] = load_table(r, cfg)
        except FileNotFoundError:
            log.warning("missing table %s", r)
    report: dict[str, Any] = {
        "zones": {z: zone_report(z, df, cfg) for z, df in tables.items()},
        "zone_sum_vs_national": zone_sum_check(tables),
        "rules": {
            "short_gap_max_hours": cfg.quality.max_interp_gap_h,
            "short_gap_treatment": "linear interpolation, flagged",
            "long_gap_treatment": "left missing; excluded from windows and evaluation",
            "outlier_rule": f"robust z of residual after yearly weekday-hour median > "
            f"{cfg.quality.outlier_z}; flagged, not removed",
        },
    }
    out = cfg.path("results") / "data_quality.json"
    out.write_text(
        json.dumps(
            report, indent=2, default=lambda o: float(o) if isinstance(o, np.generic) else str(o)
        )
    )
    return report

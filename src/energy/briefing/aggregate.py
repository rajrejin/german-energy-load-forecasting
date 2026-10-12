"""Deterministic weekly aggregation: forecasts in, a JSON-able dict of facts out."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

FEATURE_LABELS = {
    "hist:load": "recent load history",
    "hist:holiday_share": "recent holidays",
    "hist:doy_sin": "season of the year (history)",
    "hist:doy_cos": "season of the year (history)",
    "hist_cal:hour": "hour of day (history)",
    "hist_cal:weekday": "weekday pattern (history)",
    "hist_cal:holiday": "holiday flag (history)",
    "fut:holiday_share": "public holiday share",
    "fut:bridge_day": "bridge day",
    "fut:christmas_period": "Christmas period",
    "fut:doy_sin": "season of the year",
    "fut:doy_cos": "season of the year",
    "fut:lag168": "load one week earlier",
    "fut:latest_same_hour": "load at the same hour on the latest observed day",
    "fut:lag48": "load two days earlier",
    "fut:last_obs_load": "most recent observed load",
    "fut:load_mean24": "mean load of the last 24 hours",
    "fut:temp_obs_mean24": "mean temperature of the last 24 hours",
    "fut:hdd": "heating degree term",
    "fut:cdd": "cooling degree term",
    "fut_cal:hour": "hour of day",
    "fut_cal:weekday": "weekday or weekend",
    "fut_cal:holiday": "holiday flag",
}
for _v in (
    "temperature_2m",
    "apparent_temperature",
    "relative_humidity_2m",
    "cloud_cover",
    "shortwave_radiation",
    "wind_speed_10m",
    "wind_speed_100m",
    "precipitation",
):
    FEATURE_LABELS[f"hist:obs_{_v}"] = f"recent {_v.replace('_', ' ')}"
    FEATURE_LABELS[f"fut:fc_{_v}"] = f"forecast {_v.replace('_', ' ')}"


def label(feature: str) -> str:
    return FEATURE_LABELS.get(feature, feature)


@dataclass
class DayForecast:
    date: str  # local target date, YYYY-MM-DD
    p10: np.ndarray  # (24,)
    p50: np.ndarray
    p90: np.ndarray
    hours_local: np.ndarray  # local hour of each of the 24 target hours


def _week_stats(days: list[DayForecast]) -> dict[str, Any]:
    p50 = np.concatenate([d.p50 for d in days])
    width = np.concatenate([d.p90 - d.p10 for d in days])
    k = int(np.argmax(p50))
    di, hi = divmod(k, 24)
    return {
        "peak_mw": float(p50.max()),
        "peak_date": days[di].date,
        "peak_hour_local": int(days[di].hours_local[hi]),
        "peak_day_index": di,
        "peak_hour_index": hi,
        "energy_gwh": float(p50.sum() / 1000.0),
        "mean_interval_width_mw": float(width.mean()),
    }


def aggregate_zone(
    zone: str,
    days: list[DayForecast],
    prev_days: list[DayForecast] | None,
    drivers: list[dict[str, Any]] | None,
    quality_flags: list[str] | None = None,
) -> dict[str, Any]:
    cur = _week_stats(days)
    out: dict[str, Any] = {
        "zone": zone,
        "n_days": len(days),
        "week_start": days[0].date,
        "week_end": days[-1].date,
        "expected_peak_mw": round(cur["peak_mw"], 0),
        "peak_date": cur["peak_date"],
        "peak_hour_local": cur["peak_hour_local"],
        "energy_gwh": round(cur["energy_gwh"], 1),
        "mean_interval_width_mw": round(cur["mean_interval_width_mw"], 0),
        "quality_flags": quality_flags or [],
        "peak_day_index": cur["peak_day_index"],
        "peak_hour_index": cur["peak_hour_index"],
    }
    if prev_days:
        prev = _week_stats(prev_days)
        out["previous_week_energy_gwh"] = round(prev["energy_gwh"], 1)
        out["energy_change_pct"] = round(
            (cur["energy_gwh"] - prev["energy_gwh"]) / prev["energy_gwh"] * 100.0, 1
        )
        out["previous_week_peak_mw"] = round(prev["peak_mw"], 0)
    if drivers:
        out["peak_hour_drivers"] = [
            {"driver": label(d["feature"]), "attribution_mw": round(float(d["attribution_mw"]), 0)}
            for d in drivers[:3]
        ]
    return out


def aggregate_week(zones: dict[str, dict[str, Any]]) -> dict[str, Any]:
    first = next(iter(zones.values()))
    return {
        "week_start": first["week_start"],
        "week_end": first["week_end"],
        "zones": zones,
        "interval_level_pct": 80,
        "units": {"expected_peak_mw": "MW", "energy_gwh": "GWh", "attribution_mw": "MW"},
        "note": "Forecast values from the model; drivers explain the model, not physical causes.",
    }

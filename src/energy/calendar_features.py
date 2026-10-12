"""Calendar features computed in Europe/Berlin local time, from UTC timestamps.

Holiday shares use the `holidays` package per federal state, combined with approximate
population shares per control area (configs/zones.yaml).
"""

from __future__ import annotations

from functools import lru_cache

import holidays
import numpy as np
import pandas as pd

from energy.config import load_zones

TZ = "Europe/Berlin"


@lru_cache(maxsize=64)
def _state_holidays(state: str, years: tuple[int, ...]) -> frozenset:
    h = holidays.country_holidays("DE", subdiv=state, years=list(years))
    return frozenset(h.keys())


def local_index(idx_utc: pd.DatetimeIndex) -> pd.DatetimeIndex:
    return idx_utc.tz_convert(TZ)


def holiday_share(dates: pd.DatetimeIndex, states: dict[str, float]) -> np.ndarray:
    """Population-weighted share of the zone that has a public holiday on each local date."""
    years = tuple(range(int(dates.year.min()) - 1, int(dates.year.max()) + 2))
    d = np.array([x.date() for x in dates])
    tot = sum(states.values())
    share = np.zeros(len(d))
    for st, w in states.items():
        hs = _state_holidays(st, years)
        share += np.array([x in hs for x in d], dtype=float) * (w / tot)
    return share


def bridge_day(dates: pd.DatetimeIndex, hol_share: np.ndarray) -> np.ndarray:
    """Monday after a Tuesday holiday or Friday before a Saturday... (day between a holiday and
    a weekend): Friday following a Thursday holiday, Monday preceding a Tuesday holiday."""
    s = pd.Series(hol_share, index=dates.normalize())
    s = s[~s.index.duplicated()]
    prev = s.shift(1, freq="D").reindex(s.index).fillna(0.0)
    nxt = s.shift(-1, freq="D").reindex(s.index).fillna(0.0)
    wd = pd.DatetimeIndex(s.index).weekday
    fri_bridge = (wd == 4) & (prev >= 0.5)
    mon_bridge = (wd == 0) & (nxt >= 0.5)
    b = ((fri_bridge | mon_bridge) & (s < 0.5)).astype(float)
    return b.reindex(dates.normalize()).to_numpy()


def christmas_period(dates: pd.DatetimeIndex) -> np.ndarray:
    m, d = dates.month, dates.day
    return (((m == 12) & (d >= 24)) | ((m == 1) & (d <= 1))).astype(float)


def is_dst_transition(idx_utc: pd.DatetimeIndex) -> np.ndarray:
    """True for all hours on a local calendar day that has 23 or 25 hours."""
    loc_days = local_index(idx_utc).normalize().tz_localize(None)
    days = pd.DatetimeIndex(sorted(set(loc_days)))
    start = days.tz_localize(TZ)
    end = (days + pd.Timedelta(days=1)).tz_localize(TZ)
    length = (end.tz_convert("UTC") - start.tz_convert("UTC")).total_seconds() / 3600.0
    flag = pd.Series(np.asarray(length) != 24.0, index=days)
    return flag.reindex(loc_days).to_numpy()


def calendar_frame(idx_utc: pd.DatetimeIndex, zone: str) -> pd.DataFrame:
    """Calendar features for each UTC hour, evaluated in local time."""
    loc = local_index(idx_utc)
    zcfg = load_zones()["zones"]
    if zone in zcfg:
        states = zcfg[zone]["states"]
    else:  # national: weight states by sum over all zones
        states = {}
        for z in zcfg.values():
            for k, v in z["states"].items():
                states[k] = states.get(k, 0.0) + v
    # Holiday evaluation per local date, then broadcast to hours.
    days = loc.normalize().tz_localize(None)
    uniq = pd.date_range(days.min() - pd.Timedelta(days=2), days.max() + pd.Timedelta(days=2))
    hs = holiday_share(uniq, states)
    br = bridge_day(uniq, hs)
    key = pd.Series(range(len(uniq)), index=uniq)
    pos = key.reindex(days).to_numpy()
    out = pd.DataFrame(
        {
            "hour": loc.hour,
            "weekday": loc.weekday,
            "month": loc.month,
            "dayofyear": loc.dayofyear,
            "holiday_share": hs[pos],
            "bridge_day": br[pos],
            "christmas_period": christmas_period(loc),
            "is_dst_transition": is_dst_transition(idx_utc),
        },
        index=idx_utc,
    )
    out["is_weekend"] = (out["weekday"] >= 5).astype(float)
    out["is_holiday"] = (out["holiday_share"] >= 0.5).astype(float)
    return out


def cyclical(values: np.ndarray, period: float) -> tuple[np.ndarray, np.ndarray]:
    ang = 2 * np.pi * values / period
    return np.sin(ang), np.cos(ang)

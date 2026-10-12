"""Forecast origins: one per local day at the issue time, targets are the next 24 hours.

Position arithmetic works on the complete hourly UTC index of a zone table. For origin
day D (local) with issue time 10:00 local:

* ``issue_pos``      position of the issue instant (an hour boundary)
* ``last_obs_pos``   position of the last load hour that is complete at issue
                     (hour starting at ``issue - 1h``)
* ``tstart_pos``     position of local midnight at the start of day D+1
* targets            24 consecutive UTC hours from ``tstart_pos`` (on 23 and 25 hour days the
                     block ends one hour into D+2, respectively one hour before the end of D+1)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from energy.config import Config

TZ = "Europe/Berlin"
HORIZON = 24


def schedule(
    index: pd.DatetimeIndex,
    cfg: Config,
    start_date: str,
    end_date: str,
    offsets: tuple[int, ...] = (0,),
) -> pd.DataFrame:
    """Origins whose target day (local date of block start) lies in [start_date, end_date].

    ``offsets`` shifts issue time and target block together by whole hours. Used only for
    training augmentation; evaluation always uses offset 0.
    """
    t0 = index[0]
    n = len(index)
    tdays = pd.date_range(
        start_date,
        min(pd.Timestamp(end_date), index[-1].tz_convert(TZ).tz_localize(None).normalize()),
        freq="D",
    )
    if len(tdays) == 0:
        return pd.DataFrame(columns=["date", "issue_pos", "last_obs_pos", "tstart_pos", "offset"])
    d_origin = tdays - pd.Timedelta(days=1)
    issue = (d_origin + pd.Timedelta(hours=cfg.issue_hour_local)).tz_localize(TZ).tz_convert("UTC")
    tstart = tdays.tz_localize(TZ).tz_convert("UTC")
    split_end = (pd.Timestamp(end_date) + pd.Timedelta(days=1)).tz_localize(TZ).tz_convert("UTC")
    one = pd.Timedelta(hours=1)
    frames = []
    for off in offsets:
        ipos = np.asarray((issue - t0) // one) + off
        spos = np.asarray((tstart - t0) // one) + off
        end_pos = int((split_end - t0) // one)
        ok = (spos + HORIZON - 1 < min(n, end_pos)) & (ipos - 1 >= 0)
        frames.append(
            pd.DataFrame(
                {
                    "date": tdays[ok],
                    "issue_pos": ipos[ok],
                    "last_obs_pos": ipos[ok] - 1,
                    "tstart_pos": spos[ok],
                    "offset": off,
                }
            )
        )
    out = pd.concat(frames, ignore_index=True)
    return out.sort_values(["date", "offset"]).reset_index(drop=True)


def block_valid(nan_cum: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """True where there is no NaN in positions [lo, hi] inclusive; nan_cum is cumsum of isnan
    with a leading zero."""
    return (nan_cum[hi + 1] - nan_cum[lo]) == 0


def nan_cumsum(a: np.ndarray) -> np.ndarray:
    return np.concatenate([[0], np.cumsum(np.isnan(a))])

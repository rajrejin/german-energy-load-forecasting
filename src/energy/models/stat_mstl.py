"""MSTL (daily and weekly seasonality) with an ARIMA trend component, via statsforecast.

Refit schedule: the whole model (decomposition parameters and the ARIMA order search) is
refit on the trailing window every ``refit_days`` days. On the other days the fitted ARIMA
parameters are re-applied to the freshly decomposed window with ``forward``. The model uses
load history only (no weather), as a classical univariate benchmark.
"""

from __future__ import annotations

import numpy as np

from energy.features import Origins, ZoneFrame
from energy.models.base import Forecaster, empirical_residual_quantiles


def _interp(y: np.ndarray) -> np.ndarray:
    """Linear interpolation of NaNs (the validity mask already restricts gaps)."""
    if not np.isnan(y).any():
        return y
    idx = np.arange(len(y))
    ok = ~np.isnan(y)
    return np.interp(idx, idx[ok], y[ok])


class MSTLARIMA(Forecaster):
    name = "mstl_arima"

    def __init__(self, window_hours: int = 1344, refit_days: int = 7):
        self.window = window_hours
        self.refit_days = refit_days
        self._resid: dict[float, np.ndarray] | None = None

    def fit(self, zf: ZoneFrame, train: Origins, val: Origins | None = None) -> MSTLARIMA:
        return self

    def predict(self, zf: ZoneFrame, o: Origins) -> np.ndarray:
        from statsforecast.models import MSTL, AutoARIMA

        n = len(o)
        out = np.full((n, 24), np.nan)
        fitted = None
        last_fit = None
        sched = o.sched
        for i in range(n):
            if not o.valid[i]:
                continue
            last = int(sched["last_obs_pos"].iloc[i])
            tstart = int(sched["tstart_pos"].iloc[i])
            date = sched["date"].iloc[i]
            y = _interp(zf.load[last - self.window + 1 : last + 1].astype(float))
            if fitted is None or (date - last_fit).days >= self.refit_days:  # type: ignore[operator]
                m = MSTL(
                    season_length=[24, 168],
                    trend_forecaster=AutoARIMA(season_length=1, max_p=3, max_q=3),
                )
                fitted = m.fit(y)
                last_fit = date
            h = tstart + 23 - last
            f = fitted.forward(y, h=h)["mean"]
            s = tstart - last - 1
            out[i] = f[s : s + 24]
        return out

    def calibrate(self, y: np.ndarray, p: np.ndarray) -> None:
        """Store residual quantiles from validation predictions for interval output."""
        m = ~np.isnan(p).any(axis=1)
        self._resid = empirical_residual_quantiles(y[m], p[m], (0.1, 0.5, 0.9))

    def predict_quantiles(
        self, zf: ZoneFrame, o: Origins, quantiles: tuple[float, ...] = (0.1, 0.5, 0.9)
    ) -> dict[float, np.ndarray]:
        if self._resid is None:
            raise NotImplementedError("calibrate on validation first")
        p = self.predict(zf, o)
        return {q: p + self._resid[q][None, :] for q in quantiles}

"""Seasonal naive baselines."""

from __future__ import annotations

import numpy as np

from energy.features import Origins, ZoneFrame
from energy.models.base import Forecaster, empirical_residual_quantiles


class SeasonalNaive(Forecaster):
    """variant 'week': same hour one week earlier (always observed at issue time).
    variant 'day': latest observed value at the same clock hour (24 h earlier where it is
    already observed at issue time, otherwise 48 h earlier)."""

    def __init__(self, variant: str = "week"):
        if variant not in ("week", "day"):
            raise ValueError(variant)
        self.variant = variant
        self.name = f"naive_{variant}"
        self._resid: dict[float, np.ndarray] | None = None

    def fit(self, zf: ZoneFrame, train: Origins, val: Origins | None = None) -> SeasonalNaive:
        ref = val if val is not None else train
        m = ref.valid
        p = self.predict(zf, ref)
        self._resid = empirical_residual_quantiles(ref.y[m], p[m], (0.1, 0.5, 0.9))
        return self

    def predict(self, zf: ZoneFrame, o: Origins) -> np.ndarray:
        key = "lag168" if self.variant == "week" else "latest_same_hour"
        return o.fut[key].copy()

    def predict_quantiles(
        self, zf: ZoneFrame, o: Origins, quantiles: tuple[float, ...] = (0.1, 0.5, 0.9)
    ) -> dict[float, np.ndarray]:
        if self._resid is None:
            raise RuntimeError("fit first")
        p = self.predict(zf, o)
        return {q: p + self._resid[q][None, :] for q in quantiles}

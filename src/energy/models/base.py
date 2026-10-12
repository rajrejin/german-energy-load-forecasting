"""Common forecaster interface: fit, predict, predict_quantiles."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from energy.features import Origins, ZoneFrame


class Forecaster(ABC):
    name: str = "base"

    @abstractmethod
    def fit(
        self, zf: ZoneFrame, train: Origins, val: Origins | None = None
    ) -> Forecaster:  # pragma: no cover
        ...

    @abstractmethod
    def predict(self, zf: ZoneFrame, o: Origins) -> np.ndarray:  # pragma: no cover
        """Point forecast of shape (n_origins, 24)."""

    def predict_quantiles(
        self, zf: ZoneFrame, o: Origins, quantiles: tuple[float, ...] = (0.1, 0.5, 0.9)
    ) -> dict[float, np.ndarray]:
        raise NotImplementedError(f"{self.name} has no probabilistic output")


def empirical_residual_quantiles(
    y: np.ndarray, p: np.ndarray, quantiles: tuple[float, ...]
) -> dict[float, np.ndarray]:
    """Per-horizon empirical quantiles of the residual y - p, for wrapping point forecasts."""
    r = y - p
    return {q: np.nanquantile(r, q, axis=0) for q in quantiles}

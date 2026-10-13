"""Point and probabilistic forecast metrics. All arrays are (n_origins, 24) unless noted."""

from __future__ import annotations

import numpy as np


def mae(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.nanmean(np.abs(y - p)))


def rmse(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.sqrt(np.nanmean((y - p) ** 2)))


def mape(y: np.ndarray, p: np.ndarray) -> float:
    """Mean absolute percentage error in percent."""
    return float(np.nanmean(np.abs((y - p) / y)) * 100.0)


def smape(y: np.ndarray, p: np.ndarray) -> float:
    """Symmetric MAPE in percent: mean of 2|y-p| / (|y|+|p|)."""
    return float(np.nanmean(2.0 * np.abs(y - p) / (np.abs(y) + np.abs(p))) * 100.0)


def bias(y: np.ndarray, p: np.ndarray) -> float:
    """Mean error, forecast minus actual."""
    return float(np.nanmean(p - y))


def mase_scale(train_load: np.ndarray, period: int = 168) -> float:
    """In-sample MAE of the seasonal naive forecast with the given period."""
    d = np.abs(train_load[period:] - train_load[:-period])
    return float(np.nanmean(d))


def mase(y: np.ndarray, p: np.ndarray, scale: float) -> float:
    return mae(y, p) / scale


def peak_error(y: np.ndarray, p: np.ndarray) -> float:
    """Mean absolute error at the actual daily maximum hour of each origin."""
    k = np.nanargmax(y, axis=1)
    r = np.arange(len(y))
    return float(np.nanmean(np.abs(y[r, k] - p[r, k])))


def pinball(y: np.ndarray, q: np.ndarray, tau: float) -> float:
    d = y - q
    return float(np.nanmean(np.maximum(tau * d, (tau - 1.0) * d)))


def coverage(y: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    return float(np.nanmean((y >= lo) & (y <= hi)))


def interval_width(lo: np.ndarray, hi: np.ndarray) -> float:
    return float(np.nanmean(hi - lo))


def point_metrics(y: np.ndarray, p: np.ndarray, scale: float | None = None) -> dict[str, float]:
    out = {
        "mae": mae(y, p),
        "rmse": rmse(y, p),
        "mape": mape(y, p),
        "smape": smape(y, p),
        "bias": bias(y, p),
        "peak_mae": peak_error(y, p),
    }
    if scale is not None:
        out["mase"] = mase(y, p, scale)
    return out


def quantile_metrics(
    y: np.ndarray, qs: dict[float, np.ndarray], lo: float = 0.1, hi: float = 0.9
) -> dict[str, float]:
    out = {f"pinball_{t:g}": pinball(y, q, t) for t, q in qs.items()}
    if lo in qs and hi in qs:
        out["coverage_80"] = coverage(y, qs[lo], qs[hi])
        out["width_80"] = interval_width(qs[lo], qs[hi])
        out["crossing_rate"] = float(np.nanmean(qs[lo] > qs[hi]))
    return out

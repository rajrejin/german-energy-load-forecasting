"""Diebold-Mariano test with the Harvey-Leybourne-Newbold correction and block bootstrap."""

from __future__ import annotations

import math

import numpy as np
from scipy import stats


def dm_test(loss_a: np.ndarray, loss_b: np.ndarray, h: int = 1) -> dict[str, float]:
    """Two-sided DM test on the loss differential d = loss_a - loss_b.

    A negative mean means model A has lower loss. Uses the rectangular-kernel long-run
    variance with h-1 autocovariances and the Harvey et al. (1997) small-sample correction,
    compared with a Student t distribution with T-1 degrees of freedom.
    """
    d = np.asarray(loss_a, dtype=float) - np.asarray(loss_b, dtype=float)
    d = d[~np.isnan(d)]
    T = len(d)
    dbar = d.mean()
    gamma0 = np.mean((d - dbar) ** 2)
    var = gamma0
    for k in range(1, h):
        cov = np.mean((d[k:] - dbar) * (d[:-k] - dbar))
        var += 2.0 * cov
    var = var / T
    if var <= 0:
        return {"stat": float("nan"), "p_value": float("nan"), "mean_diff": float(dbar), "n": T}
    dm = dbar / math.sqrt(var)
    corr = math.sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T)
    stat = dm * corr
    p = 2.0 * (1.0 - stats.t.cdf(abs(stat), df=T - 1))
    return {"stat": float(stat), "p_value": float(p), "mean_diff": float(dbar), "n": T}


def block_bootstrap_ci(
    diff: np.ndarray,
    block: int = 7,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Moving-block bootstrap CI for the mean of a daily series (e.g. loss differences)."""
    x = np.asarray(diff, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    block = max(1, min(block, n))  # a series shorter than one block is resampled as one block
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    starts_max = n - block
    means = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, starts_max + 1, size=n_blocks)
        sample = np.concatenate([x[s : s + block] for s in starts])[:n]
        means[b] = sample.mean()
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(x.mean()), float(lo), float(hi)

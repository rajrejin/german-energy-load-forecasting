"""Interval calibration on the validation split (split conformalised quantile regression)."""

from __future__ import annotations

import numpy as np


def cqr_offset(y: np.ndarray, lo: np.ndarray, hi: np.ndarray, alpha: float = 0.2) -> float:
    """Constant Q such that [lo - Q, hi + Q] has >= 1 - alpha coverage on the calibration data.

    Nonconformity score s = max(lo - y, y - hi); Q is the ceil((n+1)(1-alpha))/n empirical
    quantile (Romano et al., 2019). Q may be negative when the raw interval is too wide.
    """
    s = np.maximum(lo - y, y - hi).ravel()
    s = s[~np.isnan(s)]
    n = len(s)
    k = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(s, k, method="higher"))


def apply_offset(lo: np.ndarray, hi: np.ndarray, q: float) -> tuple[np.ndarray, np.ndarray]:
    return lo - q, hi + q

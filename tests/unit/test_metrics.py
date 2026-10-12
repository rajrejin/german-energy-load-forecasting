"""Metric arithmetic against hand-computed values (tiny hand-written inputs)."""

import math

import numpy as np
import pytest

from energy.evaluation import metrics as M
from energy.evaluation.significance import block_bootstrap_ci, dm_test

Y = np.array([[100.0, 200.0]])
P = np.array([[110.0, 180.0]])


def test_point_metrics_hand_computed():
    assert M.mae(Y, P) == pytest.approx(15.0)
    assert M.rmse(Y, P) == pytest.approx(math.sqrt(250.0))
    assert M.mape(Y, P) == pytest.approx(10.0)
    # (2*10/210 + 2*20/380) / 2 * 100
    assert M.smape(Y, P) == pytest.approx((20 / 210 + 40 / 380) / 2 * 100)
    assert M.bias(Y, P) == pytest.approx(-5.0)
    assert M.peak_error(Y, P) == pytest.approx(20.0)
    assert M.mase(Y, P, 5.0) == pytest.approx(3.0)


def test_mase_scale_is_seasonal_naive_mae():
    x = np.array([1.0, 2.0, 3.0, 5.0, 7.0, 4.0])
    # period 3: |5-1|, |7-2|, |4-3| = 4,5,1 -> mean 10/3
    assert M.mase_scale(x, period=3) == pytest.approx(10 / 3)


def test_pinball_and_coverage():
    assert M.pinball(Y, np.array([[90.0, 210.0]]), 0.9) == pytest.approx(5.0)
    lo, hi = np.array([[90.0, 190.0]]), np.array([[110.0, 195.0]])
    assert M.coverage(Y, lo, hi) == pytest.approx(0.5)
    assert M.interval_width(lo, hi) == pytest.approx(12.5)


def test_nan_are_ignored():
    y = np.array([[1.0, np.nan]])
    p = np.array([[2.0, 5.0]])
    assert M.mae(y, p) == pytest.approx(1.0)


def test_dm_hand_computed():
    # d = a - b = [1,2,3,2,2]; mean 2; gamma0 = 0.4; var of mean = 0.08;
    # DM = 2/sqrt(0.08); Harvey factor sqrt((T+1-2h+h(h-1)/T)/T) = sqrt(4/5)
    a = np.array([2.0, 3.0, 4.0, 3.0, 3.0])
    b = np.ones(5)
    r = dm_test(a, b, h=1)
    assert r["stat"] == pytest.approx(2 / math.sqrt(0.08) * math.sqrt(0.8), rel=1e-9)
    assert r["mean_diff"] == pytest.approx(2.0)
    assert 0 < r["p_value"] < 0.01


def test_dm_identical_losses_is_nan():
    r = dm_test(np.ones(10), np.ones(10))
    assert math.isnan(r["stat"])


def test_block_bootstrap_constant_series_has_degenerate_ci():
    m, lo, hi = block_bootstrap_ci(np.full(40, 3.0), block=7, n_boot=50)
    assert m == lo == hi == pytest.approx(3.0)


def test_block_bootstrap_ci_contains_mean_for_noisy_series():
    x = np.array([1.0, -1.0] * 50)
    m, lo, hi = block_bootstrap_ci(x, block=7, n_boot=300)
    assert lo <= m <= hi

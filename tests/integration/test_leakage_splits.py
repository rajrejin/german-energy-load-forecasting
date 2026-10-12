"""Leakage, split and scaler tests on real cached data (skipped when the data is absent)."""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

from energy.config import load_config
from energy.data.build import processed_path
from energy.evaluation.backtest import make_context
from energy.features import ZoneFrame, build_origins
from energy.models.lstm import fit_scaler, train_end_position
from energy.windows import HORIZON

CFG = load_config()
pytestmark = pytest.mark.skipif(
    not processed_path("DE", CFG).exists(),
    reason="processed data absent: run `energy fetch` and `energy build-dataset`",
)


@pytest.fixture(scope="module")
def ctx():
    return make_context("DE", CFG)


def test_provenance_never_after_issue_time(ctx):
    """Every load-derived feature for every origin reads only positions <= last observed."""
    for split in ("train", "val", "test"):
        o = ctx.origins(split)
        last = o.sched["last_obs_pos"].to_numpy()
        for name, src in o.provenance.items():
            m = src >= 0
            lim = last[:, None] if src.ndim == 2 else last
            assert np.all(np.where(m, src, -1) <= np.broadcast_to(lim, src.shape)), name


def test_last_observed_hour_is_before_issue_instant(ctx):
    o = ctx.origins("test")
    issue = ctx.zf.index[o.sched["issue_pos"].to_numpy()]
    last = ctx.zf.index[o.sched["last_obs_pos"].to_numpy()]
    assert np.all(last + pd.Timedelta(hours=1) <= issue)
    assert np.all(issue.tz_convert("Europe/Berlin").hour == CFG.issue_hour_local)


def test_perturbing_the_future_does_not_change_features(ctx):
    """Replace all load after the last observed hour of sampled origins; features must match."""
    o = ctx.origins("test")
    rng = np.random.default_rng(0)
    for i in rng.choice(np.flatnonzero(o.valid), size=8, replace=False):
        last = int(o.sched["last_obs_pos"].iloc[i])
        df2 = ctx.zf.df.copy()
        df2.iloc[last + 1 :, df2.columns.get_loc("load")] = 1e9
        zf2 = ZoneFrame("DE", df2, CFG)
        d = o.dates[i].strftime("%Y-%m-%d")
        o2 = build_origins(zf2, CFG, d, d)
        assert len(o2) == 1
        for name, arr in o.fut.items():
            if name.startswith(("lag", "latest")) or name in ():
                a, b = arr[i], o2.fut[name][0]
                if name == "latest_same_hour":
                    assert np.allclose(a, b)
                else:
                    assert np.allclose(np.nan_to_num(a, nan=-1), np.nan_to_num(b, nan=-1)), name
        for name, arr in o.ctx.items():
            assert np.allclose(arr[i], o2.ctx[name][0]), name


def test_split_targets_never_cross_boundaries(ctx):
    idx = ctx.zf.index
    ends = {}
    for split in ("train", "val", "test"):
        o = ctx.origins(split)
        s, e = ctx.dates[split]
        lo = pd.Timestamp(s).tz_localize("Europe/Berlin").tz_convert("UTC")
        hi = (pd.Timestamp(e) + pd.Timedelta(days=1)).tz_localize("Europe/Berlin").tz_convert("UTC")
        tp = o.tpos[o.valid].ravel()
        ts = idx[tp]
        assert ts.min() >= lo and ts.max() < hi, split
        ends[split] = (ts.min(), ts.max())
    assert ends["train"][1] < ends["val"][0]
    assert ends["val"][1] < ends["test"][0]


def test_split_target_sets_are_disjoint(ctx):
    sets = {}
    for split in ("train", "val", "test"):
        o = ctx.origins(split)
        sets[split] = set(o.tpos[o.valid].ravel().tolist())
    assert not (sets["train"] & sets["val"])
    assert not (sets["val"] & sets["test"])
    assert not (sets["train"] & sets["test"])


def test_augmented_training_phases_stay_inside_train_split(ctx):
    o = ctx.origins("train", offsets=(0, 6, 12, 18))
    ts = ctx.zf.index[o.tpos[o.valid].ravel()]
    cutoff = pd.Timestamp(CFG.splits.train_end, tz="Europe/Berlin") + pd.Timedelta(days=1)
    assert ts.max() < cutoff.tz_convert("UTC")


def test_scaler_uses_training_period_only(ctx):
    zf = ctx.zf
    te = train_end_position(zf, CFG.splits.train_end)
    sc = fit_scaler(zf, te)
    assert sc.load_mean == pytest.approx(float(np.nanmean(zf.load[:te])))
    df2 = zf.df.copy()
    df2.iloc[te:, df2.columns.get_loc("load")] = 5e6
    df2.iloc[te:, df2.columns.get_loc("temperature_2m")] = 99.0
    sc2 = fit_scaler(ZoneFrame("DE", df2, CFG), te)
    assert sc2.load_mean == sc.load_mean and sc2.w_mean == sc.w_mean
    assert zf.index[te - 1] < pd.Timestamp(CFG.splits.train_end, tz="UTC") + pd.Timedelta(days=1)
    assert zf.index[te] >= pd.Timestamp(CFG.splits.train_end, tz="UTC") + pd.Timedelta(days=1)


def test_dst_days_have_correct_block_alignment(ctx):
    """Around real DST transitions the target block still starts at local midnight.

    A range of two days is used because the 24 hour block of a 23 hour day reaches into the
    next local day and is dropped when that day is outside the requested range."""
    for day in ("2024-03-31", "2024-10-27", "2023-03-26", "2023-10-29"):
        nxt = (pd.Timestamp(day) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        o = build_origins(ctx.zf, CFG, day, nxt)
        k = int(np.flatnonzero(o.dates == pd.Timestamp(day))[0])
        start = ctx.zf.index[o.tpos[k, 0]].tz_convert("Europe/Berlin")
        assert (start.hour, start.minute) == (0, 0)
        assert o.tpos.shape[1] == HORIZON
        assert o.valid[k]
        last = ctx.zf.index[o.sched["last_obs_pos"].iloc[k]].tz_convert("Europe/Berlin")
        assert last.hour == CFG.issue_hour_local - 1


def test_origin_validity_uses_no_missing_values(ctx):
    o = ctx.origins("val")
    assert not np.isnan(o.y[o.valid]).any()
    assert not np.isnan(o.fut["lag168"][o.valid]).any()
    _ = copy

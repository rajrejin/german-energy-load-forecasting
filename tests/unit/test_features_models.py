"""Feature construction, baselines and models on the real fixture slice."""

import json

import numpy as np
import pandas as pd
import pytest

from energy.evaluation import metrics as M
from energy.evaluation.backtest import collect, load_preds, pred_path, save_preds, score
from energy.evaluation.intervals import apply_offset, cqr_offset
from energy.features import ZoneFrame, build_origins, to_long_frame
from energy.models.gbm import GBMForecaster
from energy.models.lstm import LSTMForecaster, fit_scaler, train_end_position
from energy.models.naive import SeasonalNaive
from energy.training.manifest import validate_manifest, write_manifest


def test_origins_shapes_and_validity(sample_ctx):
    o = sample_ctx.origins("val")
    assert o.tpos.shape == (len(o), 24)
    assert o.valid.sum() >= 28  # May 2023 has 31 days
    assert not np.isnan(o.y[o.valid]).any()


def test_lag_features_match_raw_series(sample_ctx):
    o = sample_ctx.origins("val")
    zf = sample_ctx.zf
    i = int(np.flatnonzero(o.valid)[3])
    for h in (0, 11, 23):
        t = int(o.tpos[i, h])
        assert o.fut["lag168"][i, h] == zf.load[t - 168]
        assert o.fut["lag48"][i, h] == zf.load[t - 48]
        last = int(o.sched["last_obs_pos"].iloc[i])
        if t - 24 <= last:
            assert o.fut["lag24"][i, h] == zf.load[t - 24]
        else:
            assert np.isnan(o.fut["lag24"][i, h])


def test_latest_same_hour_is_24_or_48_hours_before_target(sample_ctx):
    o = sample_ctx.origins("val")
    zf = sample_ctx.zf
    last = o.sched["last_obs_pos"].to_numpy()
    for i in np.flatnonzero(o.valid)[:5]:
        for h in range(24):
            t = int(o.tpos[i, h])
            k = 1 if t - 24 <= last[i] else 2
            assert o.fut["latest_same_hour"][i, h] == zf.load[t - 24 * k]
            assert t - 24 * k <= last[i]


def test_long_frame_rows(sample_ctx):
    o = sample_ctx.origins("val")
    df = to_long_frame(o)
    assert len(df) == len(o) * 24
    assert set(df["h"].unique()) == set(range(24))


def test_naive_week_is_lag168(sample_ctx):
    m = SeasonalNaive("week")
    val = sample_ctx.origins("val")
    m.fit(sample_ctx.zf, sample_ctx.origins("train"), val)
    p = m.predict(sample_ctx.zf, val)
    assert np.allclose(p, val.fut["lag168"], equal_nan=True)
    pr = collect(m, sample_ctx, val)
    assert set(pr.q) == {0.1, 0.5, 0.9}
    assert M.coverage(pr.y, pr.q[0.1], pr.q[0.9]) > 0.6  # calibrated on this very period


def test_naive_day_variant_and_invalid_variant(sample_ctx):
    with pytest.raises(ValueError):
        SeasonalNaive("month")
    m = SeasonalNaive("day")
    val = sample_ctx.origins("val")
    assert np.allclose(m.predict(sample_ctx.zf, val), val.fut["latest_same_hour"])
    with pytest.raises(RuntimeError):
        m.predict_quantiles(sample_ctx.zf, val)


def test_gbm_trains_and_beats_weekly_naive_on_val(sample_ctx, sample_cfg):
    g = GBMForecaster(sample_cfg, params={"n_estimators": 60, "num_leaves": 15})
    train, val = sample_ctx.origins("train"), sample_ctx.origins("val")
    g.fit(sample_ctx.zf, train, val)
    pr = collect(g, sample_ctx, val)
    nv = SeasonalNaive("week").fit(sample_ctx.zf, train, val)
    naive = collect(nv, sample_ctx, val)
    assert M.mae(pr.y, pr.p) < M.mae(naive.y, naive.p)
    assert (pr.q[0.1] <= pr.q[0.9]).all()
    imp = g.feature_importance()
    assert imp


def test_gbm_save_writes_files(sample_ctx, sample_cfg, tmp_path):
    g = GBMForecaster(sample_cfg, params={"n_estimators": 20, "num_leaves": 7})
    val = sample_ctx.origins("val")
    g.fit(sample_ctx.zf, sample_ctx.origins("train"), val)
    g.save(tmp_path)
    assert (tmp_path / "point.txt").exists() and (tmp_path / "meta.json").exists()


def test_lstm_trains_and_roundtrips(sample_ctx, sample_cfg, tmp_path):
    f = LSTMForecaster(sample_cfg, hp={"max_epochs": 3, "hidden": 16, "batch_size": 64}, seed=0)
    train = sample_ctx.origins("train", offsets=(0, 12))
    val = sample_ctx.origins("val")
    f.fit(sample_ctx.zf, train, val)
    assert len(f.history) >= 2
    assert f.history[-1]["train_loss"] < f.history[0]["train_loss"]
    pr = collect(f, sample_ctx, val)
    assert pr.p.shape[1] == 24 and np.isfinite(pr.p).all()
    assert (pr.q[0.1] <= pr.q[0.5]).all() and (pr.q[0.5] <= pr.q[0.9]).all()
    f.save(tmp_path)
    g = LSTMForecaster.load(tmp_path, sample_cfg)
    assert np.allclose(g.predict(sample_ctx.zf, val), f.predict(sample_ctx.zf, val), atol=1e-2)


def test_lstm_ablation_switches_run(sample_ctx, sample_cfg):
    val = sample_ctx.origins("val")
    for kw in ({"use_weather": False}, {"use_cal_emb": False}, {"lookback": 72}):
        f = LSTMForecaster(sample_cfg, hp={"max_epochs": 1, "hidden": 8}, **kw)
        f.fit(sample_ctx.zf, sample_ctx.origins("train", offsets=(0,)), val)
        assert np.isfinite(f.predict(sample_ctx.zf, val)[val.valid]).all()


def test_scaler_ignores_data_after_train_end(sample_ctx, sample_cfg, sample_df):
    zf = sample_ctx.zf
    te = train_end_position(zf, sample_cfg.splits.train_end)
    a = fit_scaler(zf, te)
    df2 = sample_df.copy()
    df2.iloc[te:, df2.columns.get_loc("load")] = 9e9
    b = fit_scaler(ZoneFrame("DE", df2, sample_cfg), te)
    assert a.load_mean == b.load_mean and a.load_std == b.load_std


def test_exclusion_filter_removes_origins(sample_ctx, sample_cfg):
    f = LSTMForecaster(sample_cfg, train_exclude=[("2023-02-01", "2023-02-28")])
    o = sample_ctx.origins("train")
    o2 = f._filter_excluded(o)
    feb = (o.dates >= "2023-02-01") & (o.dates <= "2023-02-28")
    assert not o2.valid[feb].any()
    assert o2.valid.sum() < o.valid.sum()


def test_cqr_offset_achieves_target_coverage():
    # normal quantile grid (deterministic, hand-built): residuals i/399 spread over [-3, 3]
    y = np.linspace(-3, 3, 400 * 24).reshape(400, 24)
    lo, hi = np.full_like(y, -0.5), np.full_like(y, 0.5)
    q = cqr_offset(y, lo, hi, 0.2)
    lo2, hi2 = apply_offset(lo, hi, q)
    assert M.coverage(y, lo2, hi2) >= 0.8
    assert q > 0


def test_pred_roundtrip(sample_ctx, sample_cfg):
    m = SeasonalNaive("week")
    val = sample_ctx.origins("val")
    m.fit(sample_ctx.zf, sample_ctx.origins("train"), val)
    pr = collect(m, sample_ctx, val)
    path = pred_path(sample_cfg, "reanalysis", "DE", "val", "naive_week")
    save_preds(path, pr)
    back = load_preds(path)
    assert np.allclose(back.p, pr.p) and set(back.q) == {0.1, 0.5, 0.9}
    assert score(back, sample_ctx.scale)["mae"] == pytest.approx(M.mae(pr.y, pr.p))


def test_manifest_validation(tmp_path, sample_cfg):
    (tmp_path / "model.bin").write_bytes(b"abc")
    with pytest.raises(ValueError, match="missing"):
        validate_manifest(tmp_path)
    write_manifest(
        tmp_path,
        model="m",
        zone="DE",
        cfg=sample_cfg,
        data_path=tmp_path / "none",
        seed=0,
        train_period=("a", "b"),
        val_metrics={"mae": 1.0},
    )
    assert validate_manifest(tmp_path)["zone"] == "DE"
    (tmp_path / "model.bin").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="hash"):
        validate_manifest(tmp_path)
    (tmp_path / "model.bin").write_bytes(b"abc")
    m = json.loads((tmp_path / "manifest.json").read_text())
    del m["seed"]
    (tmp_path / "manifest.json").write_text(json.dumps(m))
    with pytest.raises(ValueError, match="incomplete"):
        validate_manifest(tmp_path)


def test_unknown_regime_rejected(sample_ctx):
    with pytest.raises(ValueError):
        build_origins(sample_ctx.zf, sample_ctx.cfg, "2023-05-01", "2023-05-05", regime="oracle")


def test_archived_regime_uses_forecast_weather(sample_ctx):
    a = sample_ctx.origins("val", "archived")
    r = sample_ctx.origins("val", "reanalysis")
    assert a.valid.sum() > 0
    at, rt = a.fut["w_temperature_2m"][a.valid], r.fut["w_temperature_2m"][a.valid]
    assert not np.allclose(at, rt)
    assert pd.Series(at.ravel()).corr(pd.Series(rt.ravel())) > 0.9

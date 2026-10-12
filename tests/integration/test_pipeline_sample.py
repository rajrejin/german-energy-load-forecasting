"""Training, backtest, analysis, explainability and parallel runs on the real fixture slice."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from energy.config import Config, load_config
from energy.data.build import processed_path
from energy.data.quality import run_quality
from energy.evaluation import analysis
from energy.evaluation.backtest import make_context
from energy.evaluation.runner import all_artifact_names, get_preds, read_calibration, run_backtest
from energy.explain.plots import pick_cases, run_explain
from energy.explain.sanity import additivity_gap, permutation_importance
from energy.models.lstm import LSTMForecaster
from energy.models.pooled_lstm import PooledLSTM
from energy.models.stat_mstl import MSTLARIMA
from energy.training import ablations
from energy.training.hpo import run_hpo
from energy.training.train import fit_model, load_artifact, save_artifact, train_many
from tests.conftest import DATA

FAST_LSTM = {"max_epochs": 2, "hidden": 16, "batch_size": 64}


@pytest.fixture(scope="module")
def env(sample_cfg: Config, sample_df: pd.DataFrame) -> Config:
    """Config whose root holds the fixture table at the processed path."""
    p = processed_path("DE", sample_cfg)
    p.parent.mkdir(parents=True, exist_ok=True)
    sample_df.to_parquet(p)
    return sample_cfg


@pytest.fixture(scope="module")
def trained_env(env: Config):
    ctx = make_context("DE", env)
    for kind in ("naive_week", "naive_day", "lightgbm", "lstm"):
        hp = (
            {"n_estimators": 40, "num_leaves": 7}
            if kind == "lightgbm"
            else (FAST_LSTM if kind == "lstm" else None)
        )
        model, met = fit_model(
            kind, ctx, seed=0, hp=hp, **({"phase_offsets": (0, 12)} if kind == "lstm" else {})
        )
        save_artifact(kind, ctx, model, met, 0)
    return env, ctx


def test_artifacts_roundtrip_and_backtest(trained_env):
    cfg, ctx = trained_env
    names = all_artifact_names("DE", cfg)
    assert {"naive_week", "naive_day", "lightgbm", "lstm"} <= set(names)
    for n in ("lightgbm", "lstm"):
        m = load_artifact(n, "DE", cfg)
        assert np.isfinite(m.predict(ctx.zf, ctx.origins("val"))[ctx.origins("val").valid]).all()
    run_backtest("DE", names, ("val", "test"), cfg=cfg, ctx=ctx)
    pr = get_preds(cfg, "reanalysis", "DE", "test", "lstm")
    assert pr.y.shape[1] == 24 and 0.1 in pr.q
    arch = get_preds(cfg, "archived", "DE", "val", "lightgbm")
    assert len(arch.y) > 20
    # weather-free models are stored once, under reanalysis, and read for either regime
    assert len(get_preds(cfg, "archived", "DE", "val", "naive_week").y) > 20
    assert "lstm" in read_calibration(cfg)["DE"]


def test_load_best_hp_is_used_and_int_cast(env, tmp_path):
    from energy.training.train import hpo_dir, load_best_hp

    cfg = env.model_copy(update={"root": tmp_path})
    assert load_best_hp("lstm", cfg) is None
    (hpo_dir(cfg) / "best_lstm.json").write_text(
        json.dumps({"params": {"hidden": 32.0, "lr": 0.001, "layers": 1.0}})
    )
    hp = load_best_hp("lstm", cfg)
    assert hp == {"hidden": 32, "lr": 0.001, "layers": 1} and isinstance(hp["hidden"], int)


def test_analysis_tables(trained_env):
    cfg, _ = trained_env
    tabs = analysis.ladder_tables(cfg, "test", ["naive_day", "naive_week", "lightgbm", "lstm"])
    lad = tabs["ladder"]
    assert set(lad["model"]) == {"naive_day", "naive_week", "lightgbm", "lstm"}
    row = lad[(lad.model == "lstm") & (lad.regime == "reanalysis")].iloc[0]
    assert row["mae_ci_lo"] <= row["mae"] <= row["mae_ci_hi"]
    assert 0 <= row["coverage_80"] <= 1 and "coverage_80_uncalibrated" in lad
    sig = tabs["significance"]
    assert {"naive_week", "lightgbm"} <= set(sig["baseline"])
    assert (sig["ci_lo"] <= sig["ci_hi"]).all()
    bd = tabs["breakdowns"]
    assert {"daytype", "season", "extreme", "near_dst", "year"} <= set(bd["group_type"])
    assert len(tabs["by_hour"]) == 4 * 2 * 24
    out = analysis.write_all(cfg, "test")
    assert Path(out["ladder"]).exists()
    ext = analysis.external_benchmark(cfg, "test")
    assert isinstance(ext, pd.DataFrame)  # fixture has no day-ahead forecast column values
    spread = analysis.seed_spread(cfg, "test")
    assert len(spread) == 1 and spread["n_seeds"].iloc[0] == 1


def test_date_attrs_classify_days(sample_ctx):
    dates = pd.DatetimeIndex(["2023-03-25", "2023-03-26", "2023-05-01", "2023-05-02"])
    a = analysis.date_attrs(sample_ctx.zf.df, dates)
    assert list(a["daytype"]) == ["weekend", "weekend", "holiday", "weekday"]
    assert a.loc["2023-03-26", "near_dst"] == "within 1 day of DST change"
    assert a.loc["2023-05-02", "near_dst"] == "other"


def test_quality_report_on_fixture(env):
    rep = run_quality(env)
    z = rep["zones"]["DE"]
    assert z["rows"] == 5088 and z["duplicate_timestamps"] == 0
    assert z["load_missing_after_treatment_hours"] == 0
    assert z["dst_transition_hours"] == 23  # 2023-03-26 has 23 hours
    assert rep["zone_sum_vs_national"]["available"] is False
    assert (env.root / "results" / "data_quality.json").exists()


def test_mstl_arima_predicts_reasonably(sample_ctx):
    val = sample_ctx.origins("val")
    sub = sample_ctx.origins("val")
    sub.valid = sub.valid & (np.arange(len(sub)) < 3)
    p = MSTLARIMA(window_hours=1344, refit_days=7).predict(sample_ctx.zf, sub)
    ok = sub.valid
    assert np.isfinite(p[ok]).all()
    mape = np.mean(np.abs(p[ok] - val.y[ok]) / val.y[ok]) * 100
    assert mape < 15
    m = MSTLARIMA()
    with pytest.raises(NotImplementedError):
        m.predict_quantiles(sample_ctx.zf, sub)


def test_shap_sanity_and_plots(trained_env):
    cfg, ctx = trained_env
    rep = run_explain("DE", cfg, quick=True)
    assert 0 <= rep["additivity"]["gap_relative_to_output_deviation"] < 5
    assert -1 <= rep["permutation_agreement"]["spearman_rho"] <= 1
    assert "weekend_flag" in rep["directional"]
    assert rep["stability"]["pairwise_spearman"]
    assert "cold_snap" in rep["cases"]
    figs = list((cfg.root / "results" / "figures" / "explain").glob("*.png"))
    assert any(f.name.startswith("waterfall") for f in figs) and any(
        f.name.startswith("global_importance") for f in figs
    )
    f = load_artifact("lstm", "DE", cfg)
    assert isinstance(f, LSTMForecaster)
    perm = permutation_importance(f, ctx, ctx.origins("val"), repeats=1)
    assert perm["fut:lag168"] > perm["fut:christmas_period"]
    cases = pick_cases(ctx, ctx.origins("val"))
    assert cases["cold_snap"]["date"].startswith("2023-05")
    gap = additivity_gap(
        type(
            "E",
            (),
            {"attr": np.array([[1.0, 2.0]]), "base_value": 10.0, "output": np.array([13.5])},
        )()
    )
    assert gap["mean_abs_gap_mw"] == pytest.approx(0.5)  # |1+2+10-13.5|


def test_hpo_is_resumable_and_writes_best(env, tmp_path):
    cfg = env.model_copy(update={"root": tmp_path})
    p = processed_path("DE", cfg)
    p.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(processed_path("DE", env), p)
    df = run_hpo("lightgbm", "DE", n_trials=2, cfg=cfg)
    assert len(df) == 2
    df2 = run_hpo("lightgbm", "DE", n_trials=3, cfg=cfg)  # resumes: first 2 rows kept
    assert len(df2) == 3 and np.allclose(df2["val_mae"].iloc[:2], df["val_mae"])
    best = json.loads((cfg.root / "results" / "hpo" / "best_lightgbm.json").read_text())
    assert best["params"]["num_leaves"] in (15, 31, 63, 127)
    df3 = run_hpo("lstm", "DE", n_trials=1, max_epochs=1, cfg=cfg)
    assert len(df3) == 1 and df3["select"].iloc[0] > 0


def test_train_many_is_resumable(env, tmp_path, monkeypatch):
    cfg = env.model_copy(update={"root": tmp_path})
    p = processed_path("DE", cfg)
    p.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(processed_path("DE", env), p)
    monkeypatch.setattr("energy.training.train.load_config", lambda *a, **k: cfg)
    train_many(["DE"], ["naive_week"], [0])
    m1 = (cfg.root / "artifacts" / "models" / "DE" / "naive_week" / "manifest.json").stat().st_mtime
    train_many(["DE"], ["naive_week"], [0])
    assert (
        cfg.root / "artifacts" / "models" / "DE" / "naive_week" / "manifest.json"
    ).stat().st_mtime == m1


def test_pooled_lstm_trains_on_two_views_of_fixture(env, tmp_path):
    """Plumbing test: the same real table is registered under two zone names."""
    ctx = make_context("DE", env)
    pooled = PooledLSTM(env, ["DE", "TenneT"], hp={"max_epochs": 2, "hidden": 16, "batch_size": 64})
    ctx2 = make_context("TenneT", env, df=ctx.zf.df)
    pooled.fit({"DE": ctx, "TenneT": ctx2})
    val = ctx.origins("val")
    view = pooled.view("TenneT")
    assert view.predict(ctx2.zf, val).shape == (len(val), 24)
    q = view.predict_quantiles(ctx2.zf, val)
    assert (q[0.1] <= q[0.9]).all()
    pooled.save(tmp_path)
    again = PooledLSTM.load(tmp_path, env)
    assert np.allclose(
        again.view("DE").predict(ctx.zf, val), pooled.view("DE").predict(ctx.zf, val), atol=1e-2
    )


def test_ablation_registry_is_consistent():
    assert {"lstm_abl_noweather", "lstm_abl_nocalemb", "lstm_abl_lb72", "lstm_abl_lb336"} <= set(
        ablations.ABLATIONS
    )
    assert "lightgbm_abl_laggen" in ablations.ABLATIONS
    assert not any(
        f.startswith("w_") for f in ablations.ABLATIONS["lightgbm_abl_noweather"][1]["features"]
    )


def test_ablation_run_on_fixture(env, tmp_path):
    cfg = env.model_copy(update={"root": tmp_path})
    p = processed_path("DE", cfg)
    p.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(processed_path("DE", env), p)
    ablations.run_ablations("DE", ["lightgbm_abl_noweather"], cfg=cfg)
    assert (
        cfg.root / "artifacts" / "models" / "DE" / "lightgbm_abl_noweather" / "manifest.json"
    ).exists()
    ablations.run_ablations("DE", ["lightgbm_abl_noweather"], cfg=cfg)  # skip path


def test_parallel_matches_sequential_on_fixture(env, tmp_path, monkeypatch):
    """Dask workers and the sequential reference give the same predictions."""
    from energy.config import load_config as lc
    from energy.training.parallel import PeakMemory, run_parallel, run_sequential

    root = tmp_path / "root"
    (root / "configs").mkdir(parents=True)
    shutil.copy(
        Path(__file__).parents[2] / "configs" / "zones.yaml", root / "configs" / "zones.yaml"
    )
    raw = yaml.safe_load((Path(__file__).parents[2] / "configs" / "default.yaml").read_text())
    raw["data_start"] = "2023-01-01"
    raw["splits"] = {"train_end": "2023-04-30", "val_end": "2023-05-31", "test_start": "2023-06-01"}
    (root / "configs" / "default.yaml").write_text(yaml.safe_dump(raw))
    for z in ("DE", "TenneT"):
        p = root / "data" / "processed" / f"{z}.parquet"
        p.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(DATA / "de_sample_2023.parquet", p)
    monkeypatch.setenv("ENERGY_ROOT", str(root))
    lc.cache_clear()
    out_a, out_b = tmp_path / "a", tmp_path / "b"
    out_a.mkdir()
    out_b.mkdir()
    try:
        seq = run_sequential(["DE", "TenneT"], "lightgbm", 0, 1, 1, str(out_a))
        par = run_parallel(["DE", "TenneT"], "lightgbm", 0, 1, 2, 1, "1500MB", str(out_b))
    finally:
        lc.cache_clear()
    assert seq["wall_seconds"] > 0 and par["wall_seconds"] > 0 and par["peak_rss_mb"] > 0
    for z in ("DE", "TenneT"):
        a, b = np.load(out_a / f"{z}_lightgbm.npy"), np.load(out_b / f"{z}_lightgbm.npy")
        assert np.allclose(a, b, equal_nan=True, rtol=1e-6)
    with PeakMemory(interval=0.05) as pm:
        _ = np.ones(10_000_00)
    assert pm.peak > 0
    assert load_config() is not None

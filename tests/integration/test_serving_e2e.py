"""End to end: train, publish, refresh, serve, on the real fixture slice (no network).

The refresh job normally fetches SMARD and Open-Meteo. Here the SMARD raw cache is written from
the real fixture table and the weather frames are the real fixture weather, so the code path
from `refresh` to `/v1/forecast` and `/v1/explain` runs unchanged.
"""

import gzip
import json
import time

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api.main import create_app
from api.settings import ApiSettings
from energy.data.build import WEATHER_VARS
from energy.models.lstm import LSTMForecaster
from energy.serving.forecast_service import ForecastNotFound, ForecastService
from energy.serving.refresh import publish_models, refresh
from energy.store.local import LocalStore
from energy.training.train import save_artifact

NOW = pd.Timestamp("2023-06-15 12:00", tz="UTC")


def write_smard_cache(cfg, df, upto):
    """Write the real fixture load into the SMARD raw cache format (weekly chunks)."""
    s = df["load"].loc[:upto]
    folder = cfg.path("raw") / "smard" / "410" / "DE" / "hour"
    folder.mkdir(parents=True, exist_ok=True)
    for _k, (_, wk) in enumerate(s.groupby(pd.Grouper(freq="7D", origin="start"))):
        if wk.empty:
            continue
        ts0 = int(wk.index[0].timestamp() * 1000)
        series = [[int(t.timestamp() * 1000), float(v)] for t, v in wk.items()]
        with gzip.open(folder / f"{ts0}.json.gz", "wt") as f:
            json.dump({"fetched_at": int(time.time() * 1000), "payload": {"series": series}}, f)


@pytest.fixture(scope="module")
def trained(sample_ctx, sample_cfg, tmp_path_factory):
    f = LSTMForecaster(sample_cfg, hp={"max_epochs": 2, "hidden": 16, "batch_size": 64}, seed=0)
    f.fit(sample_ctx.zf, sample_ctx.origins("train", offsets=(0, 12)), sample_ctx.origins("val"))
    save_artifact("lstm", sample_ctx, f, {"mae": 1.0}, 0)
    return f


@pytest.fixture(scope="module")
def published(trained, sample_ctx, sample_cfg, sample_df, tmp_path_factory):
    store = LocalStore(tmp_path_factory.mktemp("store"))
    publish_models(store, sample_cfg, ["DE"])
    cfg = sample_cfg.model_copy(update={"root": sample_cfg.root})
    write_smard_cache(cfg, sample_df, "2023-06-15 07:00")
    weather = {"DE": sample_df[WEATHER_VARS]}
    status = refresh(
        store, cfg, now=NOW, zones=["DE"], fetch=False, weather=weather, shap_samples=20
    )
    return store, status, weather, cfg


def test_refresh_publishes_forecast(published):
    store, status, _, _ = published
    assert status["published"] and status["zones"]["DE"]["ok"]
    fc = store.get_json("forecasts/DE/latest.json")
    assert fc["target_date"] == "2023-06-16"
    assert fc["issue_time_local"].startswith("2023-06-15T10:00")
    assert len(fc["points"]) == 24
    assert all(p["p10"] <= p["p50"] <= p["p90"] for p in fc["points"])
    assert fc["explanation_available"] and len(fc["explanations"]) == 24
    assert store.exists("actuals/DE.json") and store.exists("status/last_good.json")


def test_api_serves_published_forecast(published):
    store, *_ = published
    client = TestClient(create_app(store, ApiSettings(json_logs=False, max_forecast_age_days=5000)))
    r = client.get("/v1/forecast", params={"zone": "DE"})
    assert r.status_code == 200
    body = r.json()
    assert body["zone"] == "DE" and len(body["points"]) == 24 and body["stale"] is True
    assert r.headers["X-Request-ID"]
    r = client.get("/v1/forecast", params={"zone": "DE", "issue_time": "2023-06-15"})
    assert r.status_code == 200 and r.json()["target_date"] == "2023-06-16"
    r = client.get("/v1/explain", params={"zone": "DE", "hour": 8})
    assert r.status_code == 200
    e = r.json()
    assert e["hour_index"] == 8 and 0 < len(e["attributions"]) <= 8
    assert "not the physical causes" in e["note"]
    strict = TestClient(create_app(store, ApiSettings(json_logs=False)))  # default age bound
    r = strict.get("/v1/forecast", params={"zone": "DE", "issue_time": "2023-06-15"})
    assert r.status_code == 422 and r.json()["error"]["code"] == "invalid_issue_time"
    assert client.get("/v1/zones").json()["zones"] == ["DE"]
    assert client.get("/v1/history", params={"zone": "DE"}).status_code == 200
    ready = client.get("/v1/ready")
    assert ready.status_code == 200 and ready.json()["detail"]["zones"] == ["DE"]
    m = client.get("/metrics")
    assert "energy_api_requests_total" in m.text


def test_refresh_refuses_when_inputs_are_stale_and_api_keeps_last_good(published, sample_df):
    store, _, weather, cfg = published
    later = pd.Timestamp("2023-06-25 12:00", tz="UTC")  # load cache ends 06-15
    status = refresh(
        store, cfg, now=later, zones=["DE"], fetch=False, weather=weather, shap_samples=5
    )
    assert not status["published"] and not status["zones"]["DE"]["ok"]
    assert status["zones"]["DE"]["reasons"]
    svc = ForecastService(store, stale_after_hours=36.0)
    fc = svc.forecast("DE")
    assert fc["target_date"] == "2023-06-16"  # last good forecast still served
    refresh_status = svc.refresh_status()
    assert refresh_status is not None and refresh_status["published"] is False


def test_refresh_refuses_missing_or_tampered_artifacts(published, tmp_path):
    store, _, weather, cfg = published
    empty = LocalStore(tmp_path / "empty")
    st = refresh(empty, cfg, now=NOW, zones=["DE"], fetch=False, weather=weather)
    assert not st["published"] and "artifact problem" in st["zones"]["DE"]["reasons"][0]
    # tamper with a model file in a copy of the store
    bad = LocalStore(tmp_path / "bad")
    for k in store.list_keys("models/"):
        bad.put_bytes(k, store.get_bytes(k))
    bad.put_bytes("models/DE/lstm/model.pt", b"corrupted")
    st = refresh(bad, cfg, now=NOW, zones=["DE"], fetch=False, weather=weather)
    assert not st["published"] and "hash" in st["zones"]["DE"]["reasons"][0]


def test_service_not_found_paths(tmp_path):
    svc = ForecastService(LocalStore(tmp_path / "s"))
    for call in (
        lambda: svc.forecast("DE"),
        lambda: svc.briefing(),
        lambda: svc.history("DE"),
        lambda: svc.explain("DE", 3),
        lambda: svc.result_file("x.json"),
    ):
        with pytest.raises(ForecastNotFound):
            call()
    assert svc.zones() == [] and svc.ready()["artifacts_loaded"] is False
    assert np.isnan(np.nan)

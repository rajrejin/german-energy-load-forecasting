"""Parsers and clients against real cached API responses (served by a mock transport)."""

import gzip
import json

import httpx
import numpy as np
import pandas as pd
import pytest

from energy.data import openmeteo
from energy.data.build import _weighted_national, fill_short_gaps
from energy.data.quality import gap_runs, robust_outliers
from energy.data.smard import (
    NotAvailableError,
    SmardClient,
    load_cached_series,
    load_hourly,
    parse_chunk,
    quarterhour_to_hourly,
)
from tests.conftest import DATA


@pytest.fixture(scope="module")
def smard_payload():
    with gzip.open(DATA / "smard_410_DE_hour_chunk.json.gz", "rt") as f:
        return json.load(f)["payload"]


@pytest.fixture(scope="module")
def om_payload():
    with gzip.open(DATA / "openmeteo_archive_berlin_2023.json.gz", "rt") as f:
        return json.load(f)["payload"]


def test_parse_real_smard_chunk(smard_payload):
    s = parse_chunk(smard_payload)
    assert len(s) == 168  # one week of hourly values
    assert s.index.tz is not None and str(s.index.tz) == "UTC"
    assert s.index.is_monotonic_increasing
    assert s.dropna().between(20_000, 90_000).all()  # German grid load in MW
    assert (s.index.to_series().diff().dropna() == pd.Timedelta("1h")).all()


def test_parse_empty_and_null_values():
    assert len(parse_chunk({"series": []})) == 0
    s = parse_chunk({"series": [[1_700_000_000_000, None], [1_700_003_600_000, 5.0]]})
    assert np.isnan(s.iloc[0]) and s.iloc[1] == 5.0


def test_parse_real_openmeteo(om_payload):
    df = openmeteo.parse_hourly(om_payload)
    assert len(df) == 8760
    assert {"temperature_2m", "shortwave_radiation", "wind_speed_100m"} <= set(df.columns)
    assert df["temperature_2m"].between(-25, 42).all()
    assert (df["shortwave_radiation"] >= 0).all()
    assert df.index[0] == pd.Timestamp("2023-01-01", tz="UTC")


def test_smard_client_caches_and_parses_via_mock_transport(smard_payload, tmp_path, sample_cfg):
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        if request.url.path.endswith("index_hour.json"):
            return httpx.Response(200, json={"timestamps": [1675033200000]})
        if "404" in str(request.url):
            return httpx.Response(404)
        return httpx.Response(200, json=smard_payload)

    cfg = sample_cfg.model_copy(update={"root": tmp_path})
    cfg.smard.delay_s = 0.0
    cli = SmardClient(cfg, client=httpx.Client(transport=httpx.MockTransport(handler)))
    assert cli.fetch_index(410, "DE") == [1675033200000]
    p = cli.fetch_chunk(410, "DE", "hour", 1675033200000)
    assert p is not None and p.exists()
    n = len(calls)
    cli.fetch_chunk(410, "DE", "hour", 1675033200000)  # old chunk fetched long after: cached
    assert len(calls) in (n, n + 1)
    s = load_cached_series(410, "DE", "hour", cfg)
    assert len(s) > 100
    hourly, src = load_hourly(410, "DE", cfg)
    assert src == "hour" and len(hourly) == len(s)
    assert (
        cli.fetch_series(
            410, "DE", "hour", start="2023-01-01", end=pd.Timestamp("2023-02-15", tz="UTC")
        )
        == "ok"
    )


def test_smard_unavailable_combination_is_reported(tmp_path, sample_cfg):
    def handler(request):
        return httpx.Response(404)

    cfg = sample_cfg.model_copy(update={"root": tmp_path})
    cfg.smard.delay_s = 0.0
    cli = SmardClient(cfg, client=httpx.Client(transport=httpx.MockTransport(handler)))
    assert cli.fetch_series(1223, "TransnetBW") == "unavailable"
    with pytest.raises(NotAvailableError):
        cli.fetch_index(1223, "TransnetBW")


def test_quarterhour_aggregation_hand_computed():
    idx = pd.date_range("2024-01-01", periods=9, freq="15min", tz="UTC")
    s = pd.Series([4.0, 8.0, 12.0, 16.0, 1.0, 2.0, 3.0, 4.0, 100.0], index=idx)
    out = quarterhour_to_hourly(s)
    # hour 0: mean(4,8,12,16)=10 ; hour 1: mean(1,2,3,4)=2.5 ; hour 2 has one value: dropped
    assert list(out.values) == [10.0, 2.5]


def test_fill_short_gaps_hand_computed():
    s = pd.Series([1.0, np.nan, np.nan, 4.0, np.nan, np.nan, np.nan, np.nan, 9.0, np.nan])
    filled, flag = fill_short_gaps(s, 3)
    assert list(filled[:4]) == [1.0, 2.0, 3.0, 4.0]  # gap of 2 interpolated
    assert flag[1] and flag[2] and not flag[3]
    assert filled[4:8].isna().all()  # gap of 4 left missing
    assert np.isnan(filled.iloc[9]) and not flag.iloc[9]  # trailing gap not extrapolated


def test_gap_runs_and_outliers(sample_df):
    m = pd.Series([False, True, True, False, True], index=pd.RangeIndex(5))
    assert gap_runs(m) == [(1, 2), (4, 1)]
    load = sample_df["load"].copy()
    assert robust_outliers(load, 8.0).sum() == 0
    load.iloc[1000] = load.iloc[1000] * 3
    assert robust_outliers(load, 8.0).iloc[1000]


def test_weighted_national_renormalises_over_missing():
    idx = pd.date_range("2024-01-01", periods=2, freq="1h", tz="UTC")
    a = pd.DataFrame({"t": [10.0, np.nan]}, index=idx)
    b = pd.DataFrame({"t": [20.0, 30.0]}, index=idx)
    out = _weighted_national({"a": a, "b": b}, {"a": 3.0, "b": 1.0})
    assert out["t"].iloc[0] == pytest.approx(12.5)  # (3*10 + 1*20)/4
    assert out["t"].iloc[1] == pytest.approx(30.0)  # only b present


def test_zone_weather_population_weights():
    w = openmeteo.zone_weights("TransnetBW")
    assert sum(w.values()) == pytest.approx(1.0)
    assert w["Stuttgart"] == pytest.approx(630 / 1250)
    frames = {
        "Stuttgart": pd.DataFrame({"t": [10.0]}, index=pd.DatetimeIndex(["2024-01-01"], tz="UTC")),
        "Karlsruhe": pd.DataFrame({"t": [20.0]}, index=pd.DatetimeIndex(["2024-01-01"], tz="UTC")),
        "Mannheim": pd.DataFrame({"t": [30.0]}, index=pd.DatetimeIndex(["2024-01-01"], tz="UTC")),
    }
    out = openmeteo.zone_weather("archive", "TransnetBW", frames=frames)
    assert out["t"].iloc[0] == pytest.approx((630 * 10 + 310 * 20 + 310 * 30) / 1250)


def test_openmeteo_client_caches_year(om_payload, tmp_path, sample_cfg):
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.params.get("start_date", "live"))
        return httpx.Response(200, json=om_payload)

    cfg = sample_cfg.model_copy(update={"root": tmp_path})
    cfg.openmeteo.delay_s = 0.0
    cli = openmeteo.OpenMeteoClient(
        cfg, client=httpx.Client(transport=httpx.MockTransport(handler))
    )
    p = cli.fetch_year(openmeteo.ARCHIVE, "Berlin", 52.5, 13.4, 2023)
    assert p is not None and p.exists()
    assert calls == ["2023-01-01"]
    cli.fetch_year(openmeteo.ARCHIVE, "Berlin", 52.5, 13.4, 2023)  # old year: served from cache
    assert len(calls) == 1
    df = openmeteo.load_point(openmeteo.ARCHIVE, "Berlin", cfg)
    assert len(df) == 8760
    live = cli.fetch_live(52.5, 13.4)
    assert "hourly" in live

"""CSV importer against a real SMARD export, cross-checked with the JSON chunk parser."""

import gzip
import json

import numpy as np
import pandas as pd
import pytest

from energy.data.smard import parse_chunk
from energy.data.smard_csv import parse_export, parse_german_number
from tests.conftest import DATA


def test_german_number_arithmetic():
    assert parse_german_number("40.170,25") == 40170.25
    assert parse_german_number("1.234.567,5") == 1234567.5
    assert parse_german_number("36,7") == 36.7
    assert np.isnan(parse_german_number("-")) and np.isnan(parse_german_number(""))


def test_real_export_parses_to_utc_hourly():
    df = parse_export(DATA / "smard_export_410_DE.csv")
    assert list(df.columns) == ["Netzlast"]
    assert len(df) == 168  # one week, 2024-01-01 .. 2024-01-07 local
    assert df.index[0] == pd.Timestamp("2023-12-31 23:00", tz="UTC")  # 00:00 CET
    assert (df.index.to_series().diff().dropna() == pd.Timedelta("1h")).all()
    assert df["Netzlast"].iloc[0] == 40170.25
    assert df["Netzlast"].between(25_000, 80_000).all()


def test_csv_and_json_parsers_agree_on_real_data():
    """The same hours from the export and from the chunk endpoint must be identical."""
    csv = parse_export(DATA / "smard_export_410_DE.csv")["Netzlast"]
    from energy.config import load_config
    from energy.data.smard import load_cached_series

    cached = load_cached_series(410, "DE", "hour", load_config())
    if cached.empty:
        pytest.skip("raw SMARD cache absent (run `energy fetch`)")
    common = csv.index.intersection(cached.index)
    assert len(common) == 168
    assert np.allclose(csv.loc[common], cached.loc[common], atol=0.01)


def test_chunk_fixture_matches_known_value():
    with gzip.open(DATA / "smard_410_DE_hour_chunk.json.gz", "rt") as f:
        s = parse_chunk(json.load(f)["payload"])
    assert s.index[0] == pd.Timestamp("2023-01-29 23:00", tz="UTC")

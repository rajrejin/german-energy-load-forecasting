"""Shared fixtures. All data is the real fixture slice in tests/data (see its README)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from energy.config import Config, Splits, load_config
from energy.evaluation.backtest import Context, make_context

DATA = Path(__file__).parent / "data"


@pytest.fixture(scope="session")
def sample_df() -> pd.DataFrame:
    return pd.read_parquet(DATA / "de_sample_2023.parquet")


@pytest.fixture(scope="session")
def sample_cfg(tmp_path_factory) -> Config:
    base = load_config()
    root = tmp_path_factory.mktemp("energy_root")
    return base.model_copy(
        update={
            "data_start": "2023-01-01",
            "splits": Splits(train_end="2023-04-30", val_end="2023-05-31", test_start="2023-06-01"),
            "root": root,
        }
    )


@pytest.fixture(scope="session")
def sample_ctx(sample_df, sample_cfg) -> Context:
    return make_context("DE", sample_cfg, df=sample_df)

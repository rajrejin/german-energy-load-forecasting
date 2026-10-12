"""Time-based train / validation / test split boundaries (by local target date)."""

from __future__ import annotations

import pandas as pd

from energy.config import Config

SPLITS = ("train", "val", "test")


def split_dates(cfg: Config, last_date: pd.Timestamp | None = None) -> dict[str, tuple[str, str]]:
    """Inclusive local target-date ranges per split. The test range runs to `last_date`."""
    tr_end = pd.Timestamp(cfg.splits.train_end)
    va_end = pd.Timestamp(cfg.splits.val_end)
    te_start = pd.Timestamp(cfg.splits.test_start)
    if not (tr_end < te_start and va_end < te_start and tr_end < va_end):
        raise ValueError("split dates must satisfy train_end < val_end < test_start")
    out = {
        "train": (cfg.data_start, cfg.splits.train_end),
        "val": ((tr_end + pd.Timedelta(days=1)).strftime("%Y-%m-%d"), cfg.splits.val_end),
        "test": (
            cfg.splits.test_start,
            (last_date or pd.Timestamp("2100-01-01")).strftime("%Y-%m-%d"),
        ),
    }
    return out

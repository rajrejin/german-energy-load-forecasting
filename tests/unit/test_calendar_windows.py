"""Calendar rules, DST handling and origin arithmetic (hand-computed expectations)."""

import numpy as np
import pandas as pd
import pytest

from energy.calendar_features import calendar_frame, is_dst_transition
from energy.config import load_config
from energy.windows import HORIZON, schedule


def utc_range(start, end):
    return pd.date_range(start, end, freq="1h", tz="UTC")


def test_spring_forward_local_hours_skip_two():
    idx = utc_range("2024-03-31 00:00", "2024-03-31 02:00")
    cal = calendar_frame(idx, "TenneT")
    # 00:00Z = 01:00 CET, 01:00Z = 03:00 CEST (02:00 does not exist)
    assert list(cal["hour"]) == [1, 3, 4]


def test_fall_back_repeats_local_hour():
    idx = utc_range("2024-10-27 00:00", "2024-10-27 02:00")
    cal = calendar_frame(idx, "TenneT")
    # 00:00Z = 02:00 CEST, 01:00Z = 02:00 CET (repeated hour), 02:00Z = 03:00 CET
    assert list(cal["hour"]) == [2, 2, 3]


def test_dst_day_lengths():
    for day, hours in (("2024-03-31", 23), ("2024-10-27", 25), ("2024-06-15", 24)):
        loc = pd.date_range(day, periods=24 * 3, freq="1h", tz="UTC") - pd.Timedelta(hours=24)
        flag = is_dst_transition(loc)
        cal_days = loc.tz_convert("Europe/Berlin").strftime("%Y-%m-%d")
        n = int(flag[cal_days == day].sum())
        assert n == (hours if hours != 24 else 0)
        assert (cal_days == day).sum() == hours


def test_christmas_is_holiday_everywhere():
    idx = utc_range("2024-12-25 12:00", "2024-12-25 12:00")
    for z in ("50Hertz", "Amprion", "TenneT", "TransnetBW", "DE"):
        assert calendar_frame(idx, z)["holiday_share"].iloc[0] == pytest.approx(1.0)


def test_regional_holiday_is_partial():
    # Corpus Christi 2024-05-30: holiday in BY, HE, NW, RP, SL, BW; not in NI, SH, HB
    idx = utc_range("2024-05-30 12:00", "2024-05-30 12:00")
    tenn = calendar_frame(idx, "TenneT")["holiday_share"].iloc[0]
    assert 0.0 < tenn < 1.0
    assert calendar_frame(idx, "TransnetBW")["holiday_share"].iloc[0] == pytest.approx(1.0)
    assert calendar_frame(idx, "50Hertz")["holiday_share"].iloc[0] == pytest.approx(0.0)


def test_bridge_day_after_ascension():
    # Ascension Thursday 2024-05-09 is a holiday everywhere, Friday 2024-05-10 is a bridge day
    idx = utc_range("2024-05-10 12:00", "2024-05-10 12:00")
    assert calendar_frame(idx, "Amprion")["bridge_day"].iloc[0] == 1.0
    idx2 = utc_range("2024-05-14 12:00", "2024-05-14 12:00")
    assert calendar_frame(idx2, "Amprion")["bridge_day"].iloc[0] == 0.0


def test_schedule_normal_day_positions():
    cfg = load_config()
    idx = utc_range("2024-03-25 00:00", "2024-04-05 23:00")
    s = schedule(idx, cfg, "2024-04-01", "2024-04-01")
    assert len(s) == 1
    row = s.iloc[0]
    # issue 2024-03-31 10:00 CEST = 08:00Z; target start 2024-04-01 00:00 CEST = 03-31 22:00Z
    t0 = idx[0]
    assert row.issue_pos == int(
        (pd.Timestamp("2024-03-31 08:00", tz="UTC") - t0) / pd.Timedelta("1h")
    )
    assert row.tstart_pos - row.last_obs_pos == 15
    assert row.last_obs_pos == row.issue_pos - 1


def test_schedule_dst_target_day_is_23_hours_long_block_overflows():
    cfg = load_config()
    idx = utc_range("2024-03-25 00:00", "2024-04-05 23:00")
    s = schedule(idx, cfg, "2024-03-31", "2024-04-01")
    first = s.iloc[0]
    # target day 2024-03-31 starts 03-30 23:00Z and the 24h block ends 03-31 22:00Z
    assert idx[first.tstart_pos] == pd.Timestamp("2024-03-30 23:00", tz="UTC")
    assert idx[first.tstart_pos + HORIZON - 1] == pd.Timestamp("2024-03-31 22:00", tz="UTC")
    # issue 2024-03-30 10:00 CET = 09:00Z
    assert idx[first.issue_pos] == pd.Timestamp("2024-03-30 09:00", tz="UTC")


def test_schedule_blocks_never_cross_split_end():
    cfg = load_config()
    idx = utc_range("2024-03-25 00:00", "2024-04-05 23:00")
    s = schedule(idx, cfg, "2024-03-28", "2024-03-31")
    end = idx.get_loc(pd.Timestamp("2024-03-31 22:00", tz="UTC"))
    assert (s.tstart_pos + HORIZON - 1 <= end).all()
    # 2024-03-31 has 23 hours: its 24 hour block would end one hour into 04-01, so it is dropped
    assert s["date"].max() == pd.Timestamp("2024-03-30")


def test_issue_time_is_local_ten_across_dst():
    cfg = load_config()
    idx = utc_range("2024-10-20 00:00", "2024-11-05 23:00")
    s = schedule(idx, cfg, "2024-10-22", "2024-11-02")
    local_issue = idx[s.issue_pos.to_numpy()].tz_convert("Europe/Berlin")
    assert np.all(local_issue.hour == cfg.issue_hour_local)

"""Briefing aggregation, verification and the retry/fallback loop (stub language model)."""

import json

import numpy as np
import pytest

from energy.briefing.aggregate import DayForecast, aggregate_week, aggregate_zone, label
from energy.briefing.evaluate import week_starts
from energy.briefing.llm import SYSTEM, generate, template_briefing, user_prompt
from energy.briefing.verify import candidates, numbers_in, parse_number, verify


@pytest.fixture(scope="module")
def facts(sample_ctx):
    """Facts from two real weeks of the fixture load (p10 and p90 are +/- 4 percent bands)."""
    zf = sample_ctx.zf
    start = zf.index.get_loc(zf.index[0]) + 24 * 40
    days = []
    for k in range(14):
        sl = slice(start + 24 * k, start + 24 * (k + 1))
        y = zf.load[sl]
        days.append(DayForecast(f"2023-02-{10 + k:02d}", y * 0.96, y, y * 1.04, zf.hour[sl]))
    drivers = [
        {"feature": "fut:lag168", "attribution_mw": 1234.4},
        {"feature": "fut:fc_temperature_2m", "attribution_mw": -812.6},
        {"feature": "fut_cal:weekday", "attribution_mw": 301.2},
        {"feature": "fut:hdd", "attribution_mw": 5.0},
    ]
    z = aggregate_zone("TenneT", days[7:], days[:7], drivers, ["2 load hours were interpolated"])
    z2 = aggregate_zone("Amprion", days[7:], days[:7], None)
    return aggregate_week({"TenneT": z, "Amprion": z2}), days


def test_aggregation_matches_numpy(facts):
    f, days = facts
    z = f["zones"]["TenneT"]
    cur = np.concatenate([d.p50 for d in days[7:]])
    prev = np.concatenate([d.p50 for d in days[:7]])
    assert z["expected_peak_mw"] == round(cur.max(), 0)
    assert z["energy_gwh"] == round(cur.sum() / 1000, 1)
    assert z["energy_change_pct"] == round((cur.sum() - prev.sum()) / prev.sum() * 100, 1)
    k = int(np.argmax(cur))
    assert z["peak_hour_local"] == int(days[7 + k // 24].hours_local[k % 24])
    assert z["mean_interval_width_mw"] == round(
        np.mean(np.concatenate([d.p90 - d.p10 for d in days[7:]])), 0
    )
    assert [d["driver"] for d in z["peak_hour_drivers"]][:2] == [
        label("fut:lag168"),
        label("fut:fc_temperature_2m"),
    ]
    assert len(z["peak_hour_drivers"]) == 3


def test_template_passes_its_own_verification(facts):
    f, _ = facts
    t = template_briefing(f)
    assert t.fallback_template
    assert verify(t.model_dump(), f) == []


def test_verifier_catches_wrong_numbers_names_and_drivers(facts):
    f, _ = facts
    good = template_briefing(f).model_dump()
    bad = json.loads(json.dumps(good))
    bad["zones"]["TenneT"][0] += " That is 99999 MW."
    assert any("99999" in p for p in verify(bad, f))
    bad = json.loads(json.dumps(good))
    bad["zones"]["Bavaria"] = ["x"]
    assert any("Bavaria" in p for p in verify(bad, f))
    bad = json.loads(json.dumps(good))
    bad["risks"].append("TransnetBW looks risky.")
    assert any("TransnetBW" in p for p in verify(bad, f))
    bad = json.loads(json.dumps(good))
    bad["zones"]["Amprion"].append("Driven by heating degree term.")
    assert any("heating degree term" in p for p in verify(bad, f))
    bad = json.loads(json.dumps(good))
    del bad["zones"]["Amprion"]
    assert any("missing" in p for p in verify(bad, f))


def test_verifier_accepts_rounding_unit_conversion_and_dates(facts):
    f, _ = facts
    z = f["zones"]["TenneT"]
    peak = z["expected_peak_mw"]
    b = template_briefing(f).model_dump()
    b["zones"]["TenneT"].append(
        f"Peak is about {peak / 1000:.1f} GW, i.e. {peak:,.0f} MW, on {z['peak_date']}."
    )
    b["zones"]["TenneT"].append("This covers 7 days and the top 3 drivers.")
    assert verify(b, f) == []


def test_number_parsing_hand_cases():
    assert parse_number("12,345") == 12345.0
    assert parse_number("12,345.6") == 12345.6
    assert parse_number("3.5") == 3.5
    assert parse_number("1.234") == 1234.0  # German thousands reading
    assert candidates("1.234") == [1234.0, 1.234]  # both readings are tried by the verifier
    assert candidates("12.5") == [12.5]
    assert [t for t, _ in numbers_in("Peak 41,200 MW at hour 18, change -2.5%.")] == [
        "41,200",
        "18",
        "-2.5",
    ]


class Stub:
    name = "stub"

    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts = []

    def complete(self, system, user):
        self.prompts.append(user)
        return self.replies.pop(0)


def test_generate_first_attempt_pass(facts):
    f, _ = facts
    good = json.dumps(
        {k: v for k, v in template_briefing(f).model_dump().items() if k != "fallback_template"}
    )
    b, info = generate(f, Stub([good]))
    assert info["passed_first_attempt"] and not info["fallback"] and info["attempts"] == 1
    assert not b.fallback_template


def test_generate_retries_with_explicit_instruction_then_passes(facts):
    f, _ = facts
    t = {k: v for k, v in template_briefing(f).model_dump().items() if k != "fallback_template"}
    wrong = json.loads(json.dumps(t))
    wrong["zones"]["TenneT"][0] = "Peak 12345678 MW."
    stub = Stub([json.dumps(wrong), json.dumps(t)])
    b, info = generate(f, stub)
    assert info["attempts"] == 2 and not info["passed_first_attempt"] and not info["fallback"]
    assert "12345678" in stub.prompts[1] and "failed verification" in stub.prompts[1]


def test_generate_falls_back_after_two_retries_and_on_bad_json(facts):
    f, _ = facts
    t = {k: v for k, v in template_briefing(f).model_dump().items() if k != "fallback_template"}
    wrong = json.loads(json.dumps(t))
    wrong["headline"] = "Demand hits 77777 MW"
    b, info = generate(f, Stub([json.dumps(wrong)] * 3))
    assert b.fallback_template and info["fallback"] and info["attempts"] == 3
    b, info = generate(f, Stub(["not json at all"] * 3))
    assert b.fallback_template and info["attempts"] == 3
    b, info = generate(f, None)
    assert b.fallback_template and info["backend"] == "template"


def test_prompts_are_constrained(facts):
    f, _ = facts
    assert "ONLY" in SYSTEM and "JSON" in SYSTEM
    assert "peak_day_index" not in user_prompt(f)


def test_week_starts_requires_two_full_weeks():
    dates = [f"2024-01-{d:02d}" for d in range(1, 29)]  # Monday 2024-01-01 .. Sunday 2024-01-28
    assert week_starts(dates) == ["2024-01-08", "2024-01-15", "2024-01-22"]
    assert week_starts(dates[:20]) == ["2024-01-08"]
    assert week_starts(dates[:13]) == []

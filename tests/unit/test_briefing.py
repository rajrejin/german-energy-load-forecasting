"""Briefing aggregation, verification and the retry/fallback loop (stub language model)."""

import json

import numpy as np
import pytest

from energy.briefing.aggregate import DayForecast, aggregate_week, aggregate_zone, label
from energy.briefing.evaluate import week_starts
from energy.briefing.llm import (
    SYSTEM,
    generate,
    overview_prompt,
    template_briefing,
    zone_prompt,
)
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
    """Scripted stand-in for the language model: answers each section from the template and can
    be told to answer badly for a section a number of times first."""

    name = "stub"

    def __init__(self, facts, bad=None, garbage=False):
        self.tmpl = template_briefing(facts)
        self.bad = dict(bad or {})  # section -> remaining bad answers
        self.garbage = garbage
        self.prompts = []

    def complete(self, system, user, schema):
        self.prompts.append(user)
        if "bullets" in schema["properties"]:
            section = user.split("FACTS for zone ")[1].split(" ")[0]
            good = {"bullets": self.tmpl.zones[section]}
            wrong = {"bullets": ["Peak 12345678 MW."] + self.tmpl.zones[section][1:]}
        else:
            section = "overview"
            good = {
                "headline": self.tmpl.headline,
                "risks": self.tmpl.risks,
                "attention": self.tmpl.attention,
            }
            wrong = {**good, "headline": "Demand hits 77777 MW"}
        if self.garbage:
            return "not json at all"
        if self.bad.get(section, 0) > 0:
            self.bad[section] -= 1
            return json.dumps(wrong)
        return json.dumps(good)


def test_generate_first_attempt_pass(facts):
    f, _ = facts
    b, info = generate(f, Stub(f))
    assert info["passed_first_attempt"] and not info["fallback"]
    assert info["attempts"] == len(f["zones"]) + 1  # one call per zone plus the overview
    assert not b.fallback_template and info["final_problems"] == []


def test_generate_retries_failing_section_with_explicit_instruction(facts):
    f, _ = facts
    stub = Stub(f, bad={"TenneT": 1})
    b, info = generate(f, stub)
    assert info["sections"]["TenneT"]["attempts"] == 2
    assert not info["passed_first_attempt"] and not info["fallback"]
    retry_prompt = [p for p in stub.prompts if "failed verification" in p]
    assert len(retry_prompt) == 1 and "12345678" in retry_prompt[0]
    assert not b.fallback_template


def test_generate_falls_back_per_section_after_two_retries(facts):
    f, _ = facts
    b, info = generate(f, Stub(f, bad={"overview": 5}))
    assert info["fallback_sections"] == ["overview"] and b.fallback_template
    assert info["sections"]["overview"]["attempts"] == 3
    assert b.zones["TenneT"] == template_briefing(f).zones["TenneT"]  # zones were fine
    b, info = generate(f, Stub(f, garbage=True))
    assert b.fallback_template and set(info["fallback_sections"]) == {*f["zones"], "overview"}
    b, info = generate(f, None)
    assert b.fallback_template and info["backend"] == "template"


def test_prompts_are_constrained(facts):
    f, _ = facts
    assert "ONLY" in SYSTEM and "JSON" in SYSTEM
    assert "peak_day_index" not in zone_prompt("TenneT", f["zones"]["TenneT"], f)
    assert "peak_hour_drivers" not in overview_prompt(f)


def test_week_starts_requires_two_full_weeks():
    dates = [f"2024-01-{d:02d}" for d in range(1, 29)]  # Monday 2024-01-01 .. Sunday 2024-01-28
    assert week_starts(dates) == ["2024-01-08", "2024-01-15", "2024-01-22"]
    assert week_starts(dates[:20]) == ["2024-01-08"]
    assert week_starts(dates[:13]) == []

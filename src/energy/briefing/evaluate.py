"""Build weekly facts from backtest predictions or from stored forecasts, and evaluate briefings."""

from __future__ import annotations

import json
import logging
from typing import Any

import numpy as np
import pandas as pd

from energy.briefing.aggregate import DayForecast, aggregate_week, aggregate_zone
from energy.briefing.llm import Backend, generate
from energy.config import ALL_REGIONS, Config, load_config
from energy.evaluation.backtest import Context, make_context
from energy.evaluation.runner import get_preds
from energy.explain.shap_lstm import explain
from energy.models.lstm import LSTMForecaster
from energy.store.base import ArtifactStore, KeyNotFoundError
from energy.training.train import load_artifact

log = logging.getLogger(__name__)


def _days_from_preds(ctx: Context, regime: str, split: str, cfg: Config) -> dict[str, DayForecast]:
    pr = get_preds(cfg, regime, ctx.zone, split, "lstm")
    assert pr.tpos is not None and 0.1 in pr.q
    out = {}
    for i, d in enumerate(pd.DatetimeIndex(pr.dates)):
        out[d.strftime("%Y-%m-%d")] = DayForecast(
            d.strftime("%Y-%m-%d"),
            pr.q[0.1][i],
            pr.p[i],
            pr.q[0.9][i],
            ctx.zf.hour[pr.tpos[i]],
        )
    return out


def week_starts(dates: list[str]) -> list[str]:
    """Mondays whose 7 days, and the 7 days before, are all present."""
    s = set(dates)
    out = []
    for d in sorted(s):
        t = pd.Timestamp(d)
        if t.weekday() != 0:
            continue
        need = [(t + pd.Timedelta(days=k)).strftime("%Y-%m-%d") for k in range(-7, 7)]
        if all(x in s for x in need):
            out.append(d)
    return out


def quality_flags(ctx: Context, days: list[DayForecast]) -> list[str]:
    df = ctx.zf.df
    lo = pd.Timestamp(days[0].date).tz_localize("Europe/Berlin").tz_convert("UTC")
    hi = (
        (pd.Timestamp(days[-1].date) + pd.Timedelta(days=1))
        .tz_localize("Europe/Berlin")
        .tz_convert("UTC")
    )
    w = df.loc[lo:hi]
    flags = []
    n = int(w["load_interpolated"].sum())
    if n:
        flags.append(f"{n} load hours were interpolated")
    wn = int(sum(w[c].sum() for c in w.columns if c.endswith("_interp")))
    if wn:
        flags.append(f"{wn} weather hours were interpolated")
    return flags


def available_weeks(cfg: Config, split: str = "test", regime: str = "archived") -> list[str]:
    """Mondays with two complete weeks of forecasts in every region."""
    ctxs = {z: make_context(z, cfg) for z in ALL_REGIONS}
    days = [set(_days_from_preds(ctxs[z], regime, split, cfg)) for z in ALL_REGIONS]
    return week_starts(sorted(set.intersection(*days)))


def weekly_facts_from_backtest(
    cfg: Config | None = None,
    split: str = "test",
    regime: str = "archived",
    weeks: list[str] | None = None,
    with_drivers: bool = True,
) -> list[dict[str, Any]]:
    cfg = cfg or load_config()
    ctxs = {z: make_context(z, cfg) for z in ALL_REGIONS}
    days = {z: _days_from_preds(ctxs[z], regime, split, cfg) for z in ALL_REGIONS}
    common = set.intersection(*[set(d) for d in days.values()])
    starts = weeks or week_starts(sorted(common))
    models = {}
    origins = {}
    if with_drivers:
        for z in ALL_REGIONS:
            f = load_artifact("lstm", z, cfg)
            assert isinstance(f, LSTMForecaster)
            models[z] = f
            origins[z] = ctxs[z].origins(split, regime)
    out = []
    for ws in starts:
        t = pd.Timestamp(ws)
        cur_keys = [(t + pd.Timedelta(days=k)).strftime("%Y-%m-%d") for k in range(7)]
        prev_keys = [(t + pd.Timedelta(days=k - 7)).strftime("%Y-%m-%d") for k in range(7)]
        zfacts = {}
        for z in ALL_REGIONS:
            cur = [days[z][k] for k in cur_keys]
            prev = [days[z][k] for k in prev_keys]
            drivers = None
            peak_stats = aggregate_zone(z, cur, prev, None, quality_flags(ctxs[z], cur))
            if with_drivers:
                o = origins[z]
                vi = np.flatnonzero(o.valid)
                date = cur_keys[peak_stats["peak_day_index"]]
                row = int(np.flatnonzero(o.dates[vi] == pd.Timestamp(date))[0])
                e = explain(
                    models[z],
                    ctxs[z],
                    o,
                    np.array([row]),
                    [peak_stats["peak_hour_index"]],
                    n_background=50,
                    nsamples=50,
                )
                order = np.argsort(-np.abs(e.attr[0]))[:3]
                drivers = [
                    {"feature": e.names[i], "attribution_mw": float(e.attr[0][i])} for i in order
                ]
            zfacts[z] = aggregate_zone(z, cur, prev, drivers, quality_flags(ctxs[z], cur))
        out.append(aggregate_week(zfacts))
    return out


def facts_from_store(
    store: ArtifactStore, now: pd.Timestamp | None = None
) -> dict[str, Any] | None:
    """Weekly facts from the last 14 published daily forecasts per zone (None if too few)."""
    zfacts = {}
    for z in ALL_REGIONS:
        keys = sorted(
            k for k in store.list_keys(f"forecasts/{z}/") if not k.endswith("latest.json")
        )
        if len(keys) < 14:
            return None
        recs = [store.get_json(k) for k in keys[-14:]]

        def to_day(r: dict[str, Any]) -> DayForecast:
            return DayForecast(
                r["target_date"],
                np.array([p["p10"] for p in r["points"]]),
                np.array([p["p50"] for p in r["points"]]),
                np.array([p["p90"] for p in r["points"]]),
                np.array([p["hour_local"] for p in r["points"]]),
            )

        days = [to_day(r) for r in recs]
        cur, prev = days[7:], days[:7]
        stats = aggregate_zone(z, cur, prev, None)
        peak_rec = recs[7 + stats["peak_day_index"]]
        drivers = None
        if peak_rec.get("explanation_available"):
            items = peak_rec["explanations"][str(stats["peak_hour_index"])]
            drivers = [i for i in items if i["feature"] != "_base_value"]
        flags = ["forecast was published with explanations missing"] if drivers is None else []
        zfacts[z] = aggregate_zone(z, cur, prev, drivers, flags)
    return aggregate_week(zfacts)


def evaluate_briefings(
    backend: Backend | None,
    cfg: Config | None = None,
    n_weeks: int | None = None,
    split: str = "test",
    regime: str = "archived",
) -> dict[str, Any]:
    cfg = cfg or load_config()
    all_weeks = available_weeks(cfg, split, regime)
    chosen = all_weeks
    if n_weeks is not None and n_weeks < len(all_weeks):
        idx = np.linspace(0, len(all_weeks) - 1, n_weeks).round().astype(int)
        chosen = [all_weeks[i] for i in idx]
    facts_list = weekly_facts_from_backtest(cfg, split, regime, weeks=chosen)
    rows = []
    details = []
    for facts in facts_list:
        b, info = generate(facts, backend)
        rows.append(info)
        details.append(
            {
                "week_start": facts["week_start"],
                "facts": facts,
                "briefing": b.model_dump(),
                "info": info,
            }
        )
        log.info(
            "briefing %s attempts=%d fallback=%s",
            facts["week_start"],
            info["attempts"],
            info["fallback"],
        )
    n = len(rows)
    first = sum(1 for r in rows if r.get("passed_first_attempt"))
    after = sum(1 for r in rows if not r["fallback"])
    summary = {
        "n_weeks_evaluated": n,
        "weeks_available_in_split": len(
            weekly_facts_from_backtest(cfg, split, regime, with_drivers=False)
        ),
        "backend": rows[0]["backend"] if rows else None,
        "pass_rate_first_attempt": first / n if n else None,
        "pass_rate_after_retries": after / n if n else None,
        "template_fallback_rate": 1 - after / n if n else None,
        "mean_attempts": float(np.mean([r["attempts"] for r in rows])) if rows else None,
        "mean_seconds_per_briefing": float(np.mean([r.get("seconds", 0.0) for r in rows]))
        if rows
        else None,
        "regime": regime,
        "split": split,
    }
    out_dir = cfg.path("results")
    (out_dir / "briefing_eval.json").write_text(json.dumps(summary, indent=2))
    with open(out_dir / "briefing_eval_details.jsonl", "w", encoding="utf-8") as f:
        for d in details:
            f.write(json.dumps(d) + "\n")
    if details:
        (out_dir / "briefing_example.json").write_text(json.dumps(details[-1], indent=2))
    return summary


def publish_briefing(
    store: ArtifactStore, facts: dict[str, Any], backend: Backend | None
) -> dict[str, Any]:
    b, info = generate(facts, backend)
    payload = {
        **b.model_dump(),
        "facts": facts,
        "generation": {k: v for k, v in info.items() if k not in ("sections", "final_problems")},
        "created_utc": pd.Timestamp.now(tz="UTC").isoformat(),
    }
    store.put_json("briefing/latest.json", payload)
    return payload


__all__ = ["evaluate_briefings", "facts_from_store", "publish_briefing", "KeyNotFoundError"]

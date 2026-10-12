"""Explanation plots and the `energy explain` orchestrator."""

from __future__ import annotations

import json
import logging
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from energy.config import Config, load_config
from energy.evaluation.backtest import Context, make_context
from energy.explain.sanity import global_importance, run_sanity, stability
from energy.explain.shap_lstm import Explanation, explain
from energy.features import Origins
from energy.models.lstm import LSTMForecaster
from energy.training.train import load_artifact

log = logging.getLogger(__name__)


def global_bar(e: Explanation, path, top: int = 15, title: str = "") -> None:
    gi = global_importance(e)
    order = np.argsort(-gi)[:top][::-1]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.barh([e.names[i] for i in order], gi[order], color="#0072B2")
    ax.set_xlabel("mean |attribution| (MW)")
    ax.set_title(title or "Global importance (grouped SHAP attributions)", loc="left", fontsize=11)
    ax.grid(axis="x", alpha=0.25)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def waterfall(e: Explanation, k: int, path, title: str, top: int = 9) -> None:
    a = e.attr[k]
    order = np.argsort(-np.abs(a))
    keep = order[:top]
    other = a[order[top:]].sum()
    labels = [e.names[i] for i in keep] + ["all other features"]
    vals = np.concatenate([a[keep], [other]])
    fig, ax = plt.subplots(figsize=(8, 5))
    start = e.base_value
    cum = start
    for i, v in enumerate(vals):
        ax.barh(i, v, left=cum, color="#D55E00" if v > 0 else "#0072B2")
        ax.text(max(cum, cum + v) + 80, i, f"{v:+.0f}", va="center", fontsize=8)
        cum += v
    ax.axvline(start, color="gray", lw=1, ls="--")
    ax.axvline(e.output[k], color="black", lw=1)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("MW (dashed: base value, solid: model output)")
    ax.set_title(title, loc="left", fontsize=10)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def pick_cases(ctx: Context, o: Origins) -> dict[str, dict[str, Any]]:
    """Selection rules (validation origins): winter weekday morning, summer midday, holiday,
    cold snap. Returns row index into valid origins and target hour index."""
    vi = np.flatnonzero(o.valid)
    sched = o.sched.iloc[vi].reset_index(drop=True)
    dates = pd.DatetimeIndex(sched["date"])
    y = o.y[vi]
    temp = o.fut["w_temperature_2m"][vi].mean(axis=1)
    hol = o.fut["holiday_share"][vi].mean(axis=1)
    cases: dict[str, dict[str, Any]] = {}
    wd = dates.weekday < 5
    winter = np.isin(dates.month, [12, 1, 2]) & wd & (hol < 0.05)
    if winter.any():
        k = np.flatnonzero(winter)[np.argmax(y[winter][:, 8])]
        cases["winter_weekday_morning_peak"] = {
            "row": int(k),
            "hour": 8,
            "rule": "weekday Dec-Feb with highest 08:00 load",
        }
    summer = np.isin(dates.month, [6, 7, 8]) & wd & (hol < 0.05)
    if summer.any():
        k = np.flatnonzero(summer)[np.argsort(y[summer][:, 13])[summer.sum() // 2]]
        cases["summer_midday"] = {
            "row": int(k),
            "hour": 13,
            "rule": "summer weekday with median 13:00 load",
        }
    full_hol = (hol > 0.95) & wd
    if full_hol.any():
        k = int(np.flatnonzero(full_hol)[0])
        cases["holiday"] = {
            "row": k,
            "hour": 12,
            "rule": "first weekday public holiday in all states",
        }
    k = int(np.argmin(temp))
    cases["cold_snap"] = {
        "row": k,
        "hour": 18,
        "rule": "coldest day by mean forecast temperature",
    }
    for v in cases.values():
        v["date"] = dates[v["row"]].strftime("%Y-%m-%d")
    return cases


def run_explain(zone: str = "DE", cfg: Config | None = None, quick: bool = False) -> dict[str, Any]:
    cfg = cfg or load_config()
    ctx = make_context(zone, cfg)
    f = load_artifact("lstm", zone, cfg)
    assert isinstance(f, LSTMForecaster)
    out_dir = cfg.path("results") / "figures" / "explain"
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 40 if quick else 200
    report, e, o = run_sanity(
        f, ctx, n_samples=n, n_background=30 if quick else 100, nsamples=30 if quick else 100
    )
    report["zone"] = zone
    report["stability"] = stability(
        f,
        ctx,
        n_samples=30 if quick else 100,
        n_background=30 if quick else 100,
        nsamples=30 if quick else 100,
    )
    report["checks_passed"]["stability_mean_spearman_above_0.8"] = report["stability"]["holds"]
    global_bar(
        e,
        out_dir / f"global_importance_{zone}.png",
        title=f"{zone}: mean |attribution| over {n} validation forecast days (24 h mean)",
    )
    cases = pick_cases(ctx, o)
    report["cases"] = {}
    for name, c in cases.items():
        ex = explain(f, ctx, o, np.array([c["row"]]), [c["hour"]], n_background=100, seed=0)
        title = f"{zone} {name}: {c['date']} hour {c['hour']:02d}:00 (attributions in MW)"
        waterfall(ex, 0, out_dir / f"waterfall_{zone}_{name}.png", title)
        a = ex.attr[0]
        top = np.argsort(-np.abs(a))[:5]
        report["cases"][name] = {
            **c,
            "base_value_mw": ex.base_value,
            "model_output_mw": float(ex.output[0]),
            "additivity_gap_mw": float(abs(a.sum() + ex.base_value - ex.output[0])),
            "top_features": [{"feature": ex.names[i], "attribution_mw": float(a[i])} for i in top],
        }
    report["note"] = "Attributions explain the model, not the physical causes of demand."
    (cfg.path("results") / "explain_sanity.json").write_text(json.dumps(report, indent=2))
    return report

"""Exploratory data analysis: figures in results/figures/eda and numbers in results/eda_stats.json."""

from __future__ import annotations

import json
import logging
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from energy.config import ALL_REGIONS, ZONES, Config, load_config
from energy.data.build import load_table
from energy.data.smard import FILTERS, GENERATION_FILTERS, load_cached_series

log = logging.getLogger(__name__)

# Okabe-Ito colour-blind safe palette
PAL = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#F0E442", "#000000"]
COLORS = dict(zip(ALL_REGIONS, PAL, strict=False))


def _style(ax, title: str, xlabel: str = "", ylabel: str = "") -> None:
    ax.set_title(title, fontsize=11, loc="left")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def _local(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.index = out.index.tz_convert("Europe/Berlin")
    return out


def run_eda(cfg: Config | None = None) -> dict[str, Any]:
    cfg = cfg or load_config()
    out_dir = cfg.path("results") / "figures" / "eda"
    out_dir.mkdir(parents=True, exist_ok=True)
    tabs = {}
    for r in ALL_REGIONS:
        try:
            tabs[r] = _local(load_table(r, cfg))
        except FileNotFoundError:
            log.warning("no table for %s", r)
    stats: dict[str, Any] = {}

    # 1. load profiles by hour: weekday vs weekend, and by season (DE and zones)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    de = tabs["DE"]
    for lab, mask, c in (
        ("weekday", (de.weekday < 5) & (de.holiday_share < 0.5), PAL[0]),
        ("weekend", de.weekday >= 5, PAL[1]),
        ("holiday (Mon-Fri)", (de.weekday < 5) & (de.holiday_share >= 0.5), PAL[2]),
    ):
        g = de.loc[mask].groupby("hour")["load"].mean() / 1000
        axes[0].plot(g.index, g.values, label=lab, color=c, lw=2)
        stats.setdefault("profile_peak_hour_DE", {})[lab] = int(g.idxmax())
    _style(axes[0], "National load profile by hour (local time)", "hour of day", "GW")
    axes[0].legend(frameon=False)
    season = de.index.month.map(
        lambda m: "winter" if m in (12, 1, 2) else "spring" if m in (3, 4, 5) else "summer" if m in (6, 7, 8) else "autumn"
    )
    for i, s in enumerate(["winter", "spring", "summer", "autumn"]):
        g = de[(season == s) & (de.weekday < 5)].groupby("hour")["load"].mean() / 1000
        axes[1].plot(g.index, g.values, label=s, color=PAL[i], lw=2)
    _style(axes[1], "Weekday profile by season", "hour of day", "GW")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "load_profiles.png", dpi=130)
    plt.close(fig)

    # 2. annual trend per zone
    fig, ax = plt.subplots(figsize=(8, 4))
    ann_stats = {}
    for r in tabs:
        t = tabs[r]
        a = t.groupby(t.index.year)["load"].mean()
        counts = t.groupby(t.index.year)["load"].count()
        a = a[counts > 8000]  # complete years only
        ax.plot(a.index, a.values / 1000, marker="o", label=r, color=COLORS[r])
        ann_stats[r] = {int(k): float(v) for k, v in a.items()}
    _style(ax, "Annual mean grid load (complete years)", "year", "GW")
    ax.set_yscale("log")
    ax.legend(frameon=False, ncol=3)
    fig.tight_layout()
    fig.savefig(out_dir / "annual_trend.png", dpi=130)
    plt.close(fig)
    stats["annual_mean_mw"] = ann_stats

    # 3. holiday effect: weekday holidays vs non-holiday weekdays, ratio by zone
    fig, ax = plt.subplots(figsize=(8, 4))
    hol = {}
    for r in tabs:
        t = tabs[r]
        wk = t[(t.weekday < 5) & (t.index.year >= 2015)]
        # compare within year-month to remove seasonal level differences
        nh = wk[wk.holiday_share < 0.05]
        base = nh.groupby([nh.index.year, nh.index.month])["load"].mean()
        h = wk[wk.holiday_share >= 0.95]
        hm = h.groupby([h.index.year, h.index.month])["load"].mean()
        ratio = (hm / base.reindex(hm.index)).dropna()
        hol[r] = float(ratio.mean())
    ax.bar(list(hol), [(v - 1) * 100 for v in hol.values()], color=[COLORS[r] for r in hol])
    _style(ax, "Load on weekday public holidays vs ordinary weekdays (same month)", "", "% difference")
    fig.tight_layout()
    fig.savefig(out_dir / "holiday_effect.png", dpi=130)
    plt.close(fig)
    stats["holiday_weekday_load_ratio"] = hol

    # 4. temperature relationship (daily means)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    d = de[["load", "temperature_2m"]].resample("1D").mean().dropna()
    axes[0].scatter(d["temperature_2m"], d["load"] / 1000, s=4, alpha=0.4, color=PAL[0])
    _style(axes[0], "National daily mean load vs temperature", "temperature (C)", "GW")
    corr = {}
    for r in tabs:
        dd = tabs[r][["load", "temperature_2m"]].resample("1D").mean().dropna()
        corr[r] = float(dd["load"].corr(dd["temperature_2m"]))
        wk = dd[dd.index.weekday < 5]
        corr[r + "_weekday_only"] = float(wk["load"].corr(wk["temperature_2m"]))
    axes[1].bar(list(tabs), [corr[r] for r in tabs], color=[COLORS[r] for r in tabs])
    _style(axes[1], "Correlation of daily load and temperature", "", "Pearson r")
    fig.tight_layout()
    fig.savefig(out_dir / "temperature_relation.png", dpi=130)
    plt.close(fig)
    stats["load_temperature_corr_daily"] = corr

    # 5. COVID-19 and energy crisis: weekly mean vs the mean of the other pre-period years
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    wl = de["load"].resample("W").mean() / 1000
    wl = wl[wl.index.year >= 2015]
    for ax, yr, ref_years, ttl in (
        (axes[0], 2020, [2017, 2018, 2019], "2020 (COVID-19) vs 2017-2019 mean"),
        (axes[1], 2022, [2018, 2019, 2021], "2022 (energy crisis) vs 2018, 2019, 2021 mean"),
    ):
        cur = wl[wl.index.year == yr]
        ref = pd.concat([wl[wl.index.year == y].reset_index(drop=True) for y in ref_years], axis=1).mean(axis=1)
        n = min(len(cur), len(ref))
        ax.plot(range(n), ref.iloc[:n].values, color="gray", label="reference years", lw=2)
        ax.plot(range(n), cur.values[:n], color=PAL[1], label=str(yr), lw=2)
        _style(ax, ttl, "week of year", "GW")
        ax.legend(frameon=False)
        stats[f"yearly_ratio_{yr}"] = float(cur.mean() / ref.mean())
    if True:
        mar = de["load"]
        a = mar[(mar.index >= "2020-03-23") & (mar.index < "2020-05-04")].mean()
        ref_vals = [
            mar[(mar.index >= f"{y}-03-23") & (mar.index < f"{y}-05-04")].mean() for y in (2017, 2018, 2019)
        ]
        stats["covid_lockdown_window_ratio"] = float(a / np.mean(ref_vals))
    fig.tight_layout()
    fig.savefig(out_dir / "covid_and_crisis.png", dpi=130)
    plt.close(fig)

    # 6. generation mix by zone (annual mean MW), if generation data is cached
    gen: dict[str, dict[str, float]] = {}
    for r in ZONES:
        gen[r] = {}
        for f in GENERATION_FILTERS:
            if f in (4359, 4387):
                continue
            s = load_cached_series(f, r, "hour", cfg)
            s = s[(s.index >= "2023-01-01") & (s.index < "2024-01-01")]
            if len(s) > 8000:
                gen[r][FILTERS[f]] = float(s.mean())
    if any(gen.values()):
        fig, ax = plt.subplots(figsize=(9, 4.5))
        sources = sorted({src for g in gen.values() for src in g})
        bottom = np.zeros(len(ZONES))
        for i, src in enumerate(sources):
            vals = np.array([gen[z].get(src, 0.0) for z in ZONES]) / 1000
            ax.bar(ZONES, vals, bottom=bottom, label=src, color=plt.cm.tab20(i % 20))
            bottom += vals
        _style(ax, "Mean generation by source and control area, 2023", "", "GW (annual mean)")
        ax.legend(frameon=False, fontsize=7, ncol=2, loc="upper left", bbox_to_anchor=(1.0, 1.0))
        fig.tight_layout()
        fig.savefig(out_dir / "generation_mix_2023.png", dpi=130)
        plt.close(fig)
        stats["generation_mean_mw_2023"] = gen

    (cfg.path("results") / "eda_stats.json").write_text(json.dumps(stats, indent=2))
    return stats

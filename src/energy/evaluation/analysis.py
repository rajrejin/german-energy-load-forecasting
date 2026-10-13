"""Result tables from stored predictions: ladder, significance, breakdowns, ablations.

Every number in the README and docs is read from the CSV/JSON files written here.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import numpy as np
import pandas as pd

from energy.config import ALL_REGIONS, Config, load_config
from energy.data.build import load_table
from energy.evaluation import metrics as M
from energy.evaluation.backtest import Preds, make_context, pred_path
from energy.evaluation.runner import WEATHERLESS, get_preds
from energy.evaluation.significance import block_bootstrap_ci, dm_test

log = logging.getLogger(__name__)

LADDER = ["naive_day", "naive_week", "mstl_arima", "lightgbm", "lstm", "pooled_lstm"]
SEEDS = [0, 1, 2, 3, 4]


def try_context(zone: str, cfg: Config):  # type: ignore[no-untyped-def]
    """Context for a zone, or None when its processed table is absent."""
    try:
        return make_context(zone, cfg)
    except FileNotFoundError:
        return None


def exists(cfg: Config, regime: str, zone: str, split: str, name: str) -> bool:
    reg = "reanalysis" if name in WEATHERLESS else regime
    return pred_path(cfg, reg, zone, split, name).exists()


def load_models(
    cfg: Config, regime: str, zone: str, split: str, names: list[str], calibrated: bool = True
) -> dict[str, Preds]:
    return {
        n: get_preds(cfg, regime, zone, split, n, calibrated)
        for n in names
        if exists(cfg, regime, zone, split, n)
    }


def align(preds: dict[str, Preds]) -> tuple[pd.DatetimeIndex, dict[str, Preds]]:
    """Restrict every model to the dates that all models share."""
    common: set[Any] | None = None
    for p in preds.values():
        s = set(p.dates.tolist())
        common = s if common is None else common & s
    keep = sorted(common or [])
    out = {}
    for n, p in preds.items():
        idx = {d: i for i, d in enumerate(p.dates.tolist())}
        rows = [idx[d] for d in keep]
        out[n] = Preds(
            p.dates[rows],
            p.y[rows],
            p.p[rows],
            {k: v[rows] for k, v in p.q.items()},
            p.tpos[rows] if p.tpos is not None else None,
        )
    return pd.DatetimeIndex(np.array(keep, dtype="datetime64[ns]")), out


def date_attrs(df: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Per target-date attributes from the zone table (local calendar day)."""
    loc = df.copy()
    loc.index = loc.index.tz_convert("Europe/Berlin")
    day = loc.index.normalize().tz_localize(None)
    daily = pd.DataFrame(
        {
            "temp": loc["temperature_2m"].groupby(day).mean(),
            "hol": loc["holiday_share"].groupby(day).mean(),
            "dst": loc["is_dst_transition"].groupby(day).max(),
        }
    )
    a = daily.reindex(dates)
    a["weekday"] = dates.weekday
    a["month"] = dates.month
    a["year"] = dates.year
    a["daytype"] = np.where(
        a["hol"] >= 0.5, "holiday", np.where(a["weekday"] >= 5, "weekend", "weekday")
    )
    a["season"] = a["month"].map(
        lambda m: "winter"
        if m in (12, 1, 2)
        else "spring"
        if m in (3, 4, 5)
        else "summer"
        if m in (6, 7, 8)
        else "autumn"
    )
    lo, hi = a["temp"].quantile(0.05), a["temp"].quantile(0.95)
    a["extreme"] = np.where(
        a["temp"] <= lo, "coldest 5%", np.where(a["temp"] >= hi, "hottest 5%", "other")
    )
    dst_days = daily.index[daily["dst"].astype(bool)]
    near = set()
    for d in dst_days:
        for k in (-1, 0, 1):
            near.add(d + pd.Timedelta(days=k))
    a["near_dst"] = np.where([d in near for d in dates], "within 1 day of DST change", "other")
    return a


def _stats(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    return {
        "mae": M.mae(y, p),
        "rmse": M.rmse(y, p),
        "mape": M.mape(y, p),
        "bias": M.bias(y, p),
    }


def ladder_tables(
    cfg: Config, split: str = "test", names: list[str] | None = None
) -> dict[str, pd.DataFrame]:
    names = names or LADDER
    rows, sig_rows, hour_rows, grp_rows = [], [], [], []
    for zone in ALL_REGIONS:
        ctx = try_context(zone, cfg)
        if ctx is None:
            continue
        df = ctx.zf.df
        for regime in ("reanalysis", "archived"):
            preds = load_models(cfg, regime, zone, split, names)
            if "lstm" not in preds:
                continue
            raw = load_models(cfg, regime, zone, split, names, calibrated=False)
            dates, al = align(preds)
            _, al_raw = align(raw)
            attrs = date_attrs(df, dates)
            lstm_loss = np.abs(al["lstm"].y - al["lstm"].p).mean(axis=1)
            for n, pr in al.items():
                s = M.point_metrics(pr.y, pr.p, ctx.scale)
                daily = np.abs(pr.y - pr.p).mean(axis=1)
                m, lo, hi = block_bootstrap_ci(daily)
                row: dict[str, Any] = {
                    "zone": zone,
                    "regime": regime,
                    "split": split,
                    "model": n,
                    "n_days": len(dates),
                    **s,
                    "mae_ci_lo": lo,
                    "mae_ci_hi": hi,
                }
                if pr.q:
                    q = M.quantile_metrics(pr.y, pr.q)
                    qr = M.quantile_metrics(pr.y, al_raw[n].q) if al_raw[n].q else {}
                    row.update(q)
                    row["coverage_80_uncalibrated"] = qr.get("coverage_80", np.nan)
                    row["width_80_uncalibrated"] = qr.get("width_80", np.nan)
                rows.append(row)
                if n != "lstm":
                    d = daily - lstm_loss
                    dm = dm_test(lstm_loss, daily)
                    mean, dlo, dhi = block_bootstrap_ci(-d)  # baseline minus lstm
                    sig_rows.append(
                        {
                            "zone": zone,
                            "regime": regime,
                            "split": split,
                            "baseline": n,
                            "n_days": len(dates),
                            "mae_baseline_minus_lstm": mean,
                            "ci_lo": dlo,
                            "ci_hi": dhi,
                            "dm_stat": dm["stat"],
                            "dm_p": dm["p_value"],
                            "lstm_better_with_ci": bool(dlo > 0),
                            "baseline_better_with_ci": bool(dhi < 0),
                        }
                    )
                # hour of day
                for h in range(24):
                    hour_rows.append(
                        {
                            "zone": zone,
                            "regime": regime,
                            "model": n,
                            "hour": h,
                            "mae": float(np.abs(pr.y[:, h] - pr.p[:, h]).mean()),
                            "bias": float((pr.p[:, h] - pr.y[:, h]).mean()),
                        }
                    )
                for gtype in ("daytype", "season", "extreme", "near_dst", "year"):
                    for g, idx in attrs.groupby(gtype).indices.items():
                        st = _stats(pr.y[idx], pr.p[idx])
                        grp_rows.append(
                            {
                                "zone": zone,
                                "regime": regime,
                                "model": n,
                                "group_type": gtype,
                                "group": str(g),
                                "n_days": len(idx),
                                **st,
                            }
                        )
    out = {
        "ladder": pd.DataFrame(rows),
        "significance": pd.DataFrame(sig_rows),
        "by_hour": pd.DataFrame(hour_rows),
        "breakdowns": pd.DataFrame(grp_rows),
    }
    return out


def external_benchmark(cfg: Config, split: str = "test") -> pd.DataFrame:
    """Error of the published SMARD day-ahead load forecast (filter 411) on the same origins."""
    rows = []
    for zone in ALL_REGIONS:
        ctx = try_context(zone, cfg)
        if ctx is None:
            continue
        df = ctx.zf.df
        fc = df["smard_load_forecast"].to_numpy()
        for regime in ("reanalysis",):
            preds = load_models(cfg, regime, zone, split, ["lstm", "lightgbm", "naive_week"])
            if "lstm" not in preds:
                continue
            ref = preds["lstm"]
            assert ref.tpos is not None
            ext = fc[ref.tpos]
            ok = ~np.isnan(ext).any(axis=1)
            if ok.sum() < 30:
                continue
            y = ref.y[ok]
            rows.append(
                {
                    "zone": zone,
                    "model": "smard_day_ahead_forecast (external)",
                    "n_days": int(ok.sum()),
                    **M.point_metrics(y, ext[ok], ctx.scale),
                }
            )
            for n, p in preds.items():
                rows.append(
                    {
                        "zone": zone,
                        "model": n,
                        "n_days": int(ok.sum()),
                        **M.point_metrics(y, p.p[ok], ctx.scale),
                    }
                )
    return pd.DataFrame(rows)


def seed_spread(cfg: Config, split: str = "test") -> pd.DataFrame:
    rows = []
    for zone in ALL_REGIONS:
        maes, mapes = [], []
        ctx = try_context(zone, cfg)
        if ctx is None:
            continue
        for s in SEEDS:
            name = "lstm" if s == 0 else f"lstm_s{s}"
            if not exists(cfg, "reanalysis", zone, split, name):
                continue
            pr = get_preds(cfg, "reanalysis", zone, split, name)
            maes.append(M.mae(pr.y, pr.p))
            mapes.append(M.mape(pr.y, pr.p))
        if maes:
            gb = (
                get_preds(cfg, "reanalysis", zone, split, "lightgbm")
                if exists(cfg, "reanalysis", zone, split, "lightgbm")
                else None
            )
            rows.append(
                {
                    "zone": zone,
                    "n_seeds": len(maes),
                    "mae_mean": float(np.mean(maes)),
                    "mae_std": float(np.std(maes, ddof=1)) if len(maes) > 1 else np.nan,
                    "mape_mean": float(np.mean(mapes)),
                    "mape_std": float(np.std(mapes, ddof=1)) if len(mapes) > 1 else np.nan,
                    "lightgbm_mae": M.mae(gb.y, gb.p) if gb else np.nan,
                    "mase_mean": float(np.mean(maes) / ctx.scale),
                }
            )
    return pd.DataFrame(rows)


def ablation_table(cfg: Config, split: str = "test", zone: str = "DE") -> pd.DataFrame:
    base = ["lstm"]
    abl = sorted(
        p.stem
        for p in (cfg.path("processed") / "preds" / "reanalysis" / zone / split).glob("*.npz")
        if p.stem.startswith(("lstm_abl", "lightgbm_abl", "pooled"))
    )
    rows = []
    ctx = try_context(zone, cfg)
    if ctx is None:
        return pd.DataFrame()
    preds = load_models(cfg, "reanalysis", zone, split, base + abl)
    if "lstm" not in preds:
        return pd.DataFrame()
    dates, al = align(preds)
    for n, pr in al.items():
        daily = np.abs(pr.y - pr.p).mean(axis=1)
        m, lo, hi = block_bootstrap_ci(daily)
        rows.append(
            {
                "zone": zone,
                "variant": n,
                "n_days": len(dates),
                **M.point_metrics(pr.y, pr.p, ctx.scale),
                "mae_ci_lo": lo,
                "mae_ci_hi": hi,
            }
        )
    return pd.DataFrame(rows)


def write_all(cfg: Config | None = None, split: str = "test") -> dict[str, str]:
    cfg = cfg or load_config()
    out_dir = cfg.path("results")
    written = {}
    tabs = ladder_tables(cfg, split)
    for k, df in tabs.items():
        p = out_dir / f"{k}.csv"
        df.to_csv(p, index=False)
        written[k] = str(p)
    for name, df in (
        ("external_benchmark", external_benchmark(cfg, split)),
        ("seed_spread", seed_spread(cfg, split)),
        ("ablations", ablation_table(cfg, split)),
    ):
        p = out_dir / f"{name}.csv"
        df.to_csv(p, index=False)
        written[name] = str(p)
    # copy of by_hour under the name the dashboard expects, plus a combined json summary
    (out_dir / "backtest_manifest.json").write_text(
        json.dumps({"split": split, **written}, indent=1)
    )
    return written


__all__ = ["write_all", "ladder_tables", "ALL_REGIONS", "load_table"]

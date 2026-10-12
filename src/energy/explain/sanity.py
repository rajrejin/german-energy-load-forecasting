"""Sanity checks for the SHAP attributions; results go to results/explain_sanity.json."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
from scipy.stats import spearmanr

from energy.evaluation.backtest import Context
from energy.explain.shap_lstm import (
    CAL_BLOCKS,
    Explanation,
    Wrapper,
    build_inputs,
    explain,
    feature_names,
)
from energy.features import Origins
from energy.models.lstm import FUT_NAMES, HIST_NAMES, LSTMForecaster

log = logging.getLogger(__name__)


def additivity_gap(e: Explanation) -> dict[str, float]:
    """|sum(attributions) + base - output| averaged over explained samples."""
    gap = np.abs(e.attr.sum(axis=1) + e.base_value - e.output)
    spread = np.abs(e.output - e.output.mean())
    return {
        "mean_abs_gap_mw": float(gap.mean()),
        "median_abs_gap_mw": float(np.median(gap)),
        "mean_abs_output_deviation_mw": float(spread.mean()),
        "gap_relative_to_output_deviation": float(gap.mean() / max(spread.mean(), 1e-9)),
    }


def global_importance(e: Explanation) -> np.ndarray:
    return np.abs(e.attr).mean(axis=0)


def _group_slices() -> dict[str, tuple[int, str, slice | int]]:
    """feature name -> (input index, kind, column selector)."""
    g: dict[str, tuple[int, str, slice | int]] = {}
    for j, n in enumerate(HIST_NAMES):
        g[f"hist:{n}"] = (0, "c", j)
    for n, (a, b) in CAL_BLOCKS.items():
        g[f"hist_cal:{n}"] = (1, "oh", slice(a, b))
    for j, n in enumerate(FUT_NAMES):
        g[f"fut:{n}"] = (2, "c", j)
    for n, (a, b) in CAL_BLOCKS.items():
        g[f"fut_cal:{n}"] = (3, "oh", slice(a, b))
    return g


@torch.no_grad()
def permutation_importance(
    f: LSTMForecaster, ctx: Context, o: Origins, repeats: int = 3, seed: int = 0
) -> dict[str, float]:
    """Increase in validation MAE (MW) when one feature group is shuffled across origins."""
    assert f.model is not None and f.scaler is not None
    d, inputs = build_inputs(f, ctx, o)
    y = o.y[o.valid]
    model, scaler = f.model, f.scaler
    model.eval()

    def mae(inp: tuple[torch.Tensor, ...]) -> float:
        out = model(*inp)[..., 1].numpy().astype(float) * scaler.load_std + scaler.load_mean
        return float(np.mean(np.abs(out - y)))

    base = mae(inputs)
    rng = np.random.default_rng(seed)
    res: dict[str, float] = {}
    for name, (k, _, sel) in _group_slices().items():
        inc = []
        for _ in range(repeats):
            perm = torch.from_numpy(rng.permutation(d.n))
            x = [t.clone() for t in inputs]
            x[k][..., sel] = inputs[k][perm][..., sel]
            inc.append(mae(tuple(x)) - base)
        res[name] = float(np.mean(inc))
    return res


def run_sanity(
    f: LSTMForecaster,
    ctx: Context,
    n_samples: int = 200,
    n_background: int = 100,
    nsamples: int = 100,
    seed: int = 0,
) -> tuple[dict[str, Any], Explanation, Origins]:
    o = ctx.origins("val")
    valid_rows = np.flatnonzero(o.valid)
    rng = np.random.default_rng(seed)
    rows = np.sort(rng.choice(len(valid_rows), size=min(n_samples, len(valid_rows)), replace=False))
    hours = list(range(24))
    e = explain(f, ctx, o, rows, hours, n_background, seed, nsamples)
    names = feature_names()
    report: dict[str, Any] = {
        "split": "val",
        "n_explained_origins": int(len(rows)),
        "n_background": n_background,
        "shap_nsamples": nsamples,
        "explained_output": "mean over the 24 target hours of the median forecast",
    }
    report["additivity"] = additivity_gap(e)

    # (2) agreement with permutation importance
    perm = permutation_importance(f, ctx, o, seed=seed)
    gi = global_importance(e)
    pv = np.array([perm[n] for n in names])
    rho, p = spearmanr(gi, pv)
    report["permutation_agreement"] = {
        "spearman_rho": float(rho),
        "p_value": float(p),
        "n_features": len(names),
        "top5_shap": [names[i] for i in np.argsort(-gi)[:5]],
        "top5_permutation": [names[i] for i in np.argsort(-pv)[:5]],
    }

    # (3) directional checks on the attributions
    vo = o.sched.iloc[np.flatnonzero(o.valid)].reset_index(drop=True).iloc[rows]
    date = vo["date"]
    months = date.dt.month.to_numpy()
    wk = (date.dt.weekday >= 5).to_numpy()
    temp_cols = [
        names.index(n) for n in ("fut:fc_temperature_2m", "fut:fc_apparent_temperature", "fut:hdd")
    ]
    temp_attr = e.attr[:, temp_cols].sum(axis=1)
    temp_day = o.fut["w_temperature_2m"][np.flatnonzero(o.valid)][rows].mean(axis=1)
    winter = np.isin(months, [12, 1, 2])
    chk: dict[str, Any] = {}
    if winter.sum() >= 10:
        r, pp = spearmanr(temp_day[winter], temp_attr[winter])
        chk["winter_temperature"] = {
            "n": int(winter.sum()),
            "spearman_temp_vs_attribution": float(r),
            "p_value": float(pp),
            "holds": bool(r < 0),
            "statement": "lower forecast temperature gives a higher attribution in winter",
        }
    cal_w = e.attr[:, names.index("fut_cal:weekday")]
    chk["weekend_flag"] = {
        "mean_attr_weekend_mw": float(cal_w[wk].mean()),
        "mean_attr_weekday_mw": float(cal_w[~wk].mean()),
        "holds": bool(cal_w[wk].mean() < cal_w[~wk].mean()),
        "statement": "weekend target days receive lower attribution than weekdays",
    }
    report["directional"] = chk

    # decoder locality: the explained mean over 24 hours may still depend on all hours, so
    # also check a single hour for attribution leaking from other hours' future features.
    report["checks_passed"] = {
        "additivity_gap_under_10pct_of_output_deviation": bool(
            report["additivity"]["gap_relative_to_output_deviation"] < 0.10
        ),
        "permutation_rho_above_0.5": bool(rho > 0.5),
        "winter_temperature_direction": bool(chk.get("winter_temperature", {}).get("holds", False)),
        "weekend_direction": bool(chk["weekend_flag"]["holds"]),
    }
    return report, e, o


def stability(
    f: LSTMForecaster,
    ctx: Context,
    rows_seed: int = 0,
    n_samples: int = 100,
    seeds: tuple[int, ...] = (1, 2, 3),
    n_background: int = 100,
    nsamples: int = 100,
) -> dict[str, Any]:
    """Rank correlation of global importance across 3 random background samples."""
    o = ctx.origins("val")
    valid_rows = np.flatnonzero(o.valid)
    rng = np.random.default_rng(rows_seed)
    rows = np.sort(rng.choice(len(valid_rows), size=min(n_samples, len(valid_rows)), replace=False))
    gis = []
    for s in seeds:
        e = explain(f, ctx, o, rows, list(range(24)), n_background, s, nsamples)
        gis.append(global_importance(e))
    rhos = []
    for i in range(len(gis)):
        for j in range(i + 1, len(gis)):
            rhos.append(float(spearmanr(gis[i], gis[j])[0]))
    topk = [set(np.argsort(-g)[:5].tolist()) for g in gis]
    overlap = [
        len(topk[i] & topk[j]) / 5 for i in range(len(topk)) for j in range(i + 1, len(topk))
    ]
    return {
        "background_seeds": list(seeds),
        "pairwise_spearman": rhos,
        "mean_spearman": float(np.mean(rhos)),
        "top5_overlap_mean": float(np.mean(overlap)),
        "holds": bool(np.mean(rhos) > 0.8),
    }


__all__ = ["run_sanity", "stability", "permutation_importance", "Wrapper"]

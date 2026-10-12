"""SHAP GradientExplainer for the LSTM, with attributions aggregated to feature level.

The explained scalar is the mean of the median forecast over a set of target hours (one hour
for a waterfall, all 24 for global importance). Attributions are summed over the time steps
of the input window, and calendar one-hot blocks are summed per calendar feature.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import shap
import torch
import torch.nn as nn

from energy.evaluation.backtest import Context
from energy.features import Origins
from energy.models.lstm import (
    CAT_W,
    FUT_NAMES,
    HIST_NAMES,
    N_HOUR,
    N_WD,
    LSTMForecaster,
    ZoneData,
    make_batch,
)

log = logging.getLogger(__name__)

CAL_BLOCKS = {
    "hour": (0, N_HOUR),
    "weekday": (N_HOUR, N_HOUR + N_WD),
    "holiday": (N_HOUR + N_WD, CAT_W),
}


def feature_names() -> list[str]:
    return (
        [f"hist:{n}" for n in HIST_NAMES]
        + [f"hist_cal:{n}" for n in CAL_BLOCKS]
        + [f"fut:{n}" for n in FUT_NAMES]
        + [f"fut_cal:{n}" for n in CAL_BLOCKS]
    )


class Wrapper(nn.Module):
    """Scalar output: mean median forecast (scaled units) over the chosen target hours."""

    def __init__(self, model: nn.Module, hours: list[int]):
        super().__init__()
        self.model = model
        self.hours = hours

    def forward(self, hc, hoh, fc, foh):  # type: ignore[no-untyped-def]
        out = self.model(hc, hoh, fc, foh)[..., 1]  # (B,24) median
        return out[:, self.hours].mean(dim=1, keepdim=True)


@dataclass
class Explanation:
    names: list[str]
    attr: np.ndarray  # (n, n_features) in MW (scaled back)
    base_value: float  # MW, expected output over the background
    output: np.ndarray  # (n,) model output in MW
    indices: np.ndarray  # row indices into the valid origins


def build_inputs(
    f: LSTMForecaster, ctx: Context, o: Origins
) -> tuple[ZoneData, tuple[torch.Tensor, ...]]:
    d = ZoneData(ctx.zf, o, f.scaler, use_weather=f.use_weather)  # type: ignore[arg-type]
    idx = torch.arange(d.n)
    hc, hoh, fc, foh, _ = make_batch(d, idx, f.lookback, 1, f.use_cal_emb)
    return d, (hc, hoh, fc, foh)


def aggregate(vals: list[np.ndarray]) -> np.ndarray:
    """vals: shap arrays per input (n, ..., ) -> (n, n_features)."""
    hc, hoh, fc, foh = vals
    cols = [hc.sum(axis=1)]  # (n, F_hist)
    cols.append(np.stack([hoh[..., a:b].sum(axis=(1, 2)) for a, b in CAL_BLOCKS.values()], axis=1))
    cols.append(fc.sum(axis=1))
    cols.append(np.stack([foh[..., a:b].sum(axis=(1, 2)) for a, b in CAL_BLOCKS.values()], axis=1))
    return np.concatenate(cols, axis=1)


def sample_background(
    f: LSTMForecaster, ctx: Context, n: int = 100, seed: int = 0
) -> list[torch.Tensor]:
    """Background windows sampled from the training split."""
    tr = ctx.origins("train")
    dtr, tr_in = build_inputs(f, ctx, tr)
    rng = np.random.default_rng(seed)
    bidx = torch.from_numpy(rng.choice(dtr.n, size=min(n, dtr.n), replace=False))
    return [t[bidx] for t in tr_in]


def explain_inputs(
    f: LSTMForecaster,
    inputs: list[torch.Tensor],
    bg: list[torch.Tensor],
    hours: list[int],
    nsamples: int = 100,
) -> Explanation:
    """Attributions (MW) for already assembled model inputs against a background set."""
    assert f.model is not None and f.scaler is not None
    f.model.eval()
    wrapped = Wrapper(f.model, hours)
    expl = shap.GradientExplainer(wrapped, bg)
    vals = expl.shap_values(inputs, nsamples=nsamples)
    n = inputs[0].shape[0]
    arrs = [
        np.asarray(v).reshape(n, *inputs[i].shape[1:])
        for i, v in enumerate(_per_input(vals, inputs))
    ]
    attr = aggregate(arrs) * f.scaler.load_std
    with torch.no_grad():
        out = wrapped(*inputs).numpy().ravel() * f.scaler.load_std + f.scaler.load_mean
        base = wrapped(*bg).numpy().mean() * f.scaler.load_std + f.scaler.load_mean
    return Explanation(feature_names(), attr, float(base), out, np.arange(n))


def explain(
    f: LSTMForecaster,
    ctx: Context,
    o: Origins,
    rows: np.ndarray,
    hours: list[int],
    n_background: int = 100,
    seed: int = 0,
    nsamples: int = 100,
) -> Explanation:
    """Explain the given rows (indices into the valid origins of `o`)."""
    _, inputs = build_inputs(f, ctx, o)
    bg = sample_background(f, ctx, n_background, seed)
    e = explain_inputs(f, [t[rows] for t in inputs], bg, hours, nsamples)
    e.indices = rows
    return e


def _per_input(vals, x):  # type: ignore[no-untyped-def]
    """shap returns a list per input (single output may carry a trailing output axis)."""
    out = []
    for i, v in enumerate(vals):
        a = np.asarray(v)
        if a.ndim == x[i].dim() + 1:
            a = a[..., 0]
        out.append(a)
    return out

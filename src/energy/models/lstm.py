"""PyTorch LSTM day-ahead model with calendar embeddings and quantile output.

Inputs per origin:
* history window (last ``lookback`` hours): load, observed weather, holiday share, day-of-year
  sin/cos, plus embeddings of hour, weekday, holiday flag
* known-ahead block for the 24 target hours: forecast weather, degree terms, calendar numerics,
  load lags that are observed at issue time, summary statistics, plus embeddings
Output: quantiles (0.1, 0.5, 0.9) for each of the 24 target hours, trained with pinball loss.
Calendar embeddings are linear maps of one-hot vectors (equivalent to nn.Embedding) so that
gradient-based attribution can reach them.
"""

from __future__ import annotations

import copy
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from energy.config import Config, set_seed
from energy.data.build import WEATHER_VARS
from energy.features import Origins, ZoneFrame
from energy.models.base import Forecaster

log = logging.getLogger(__name__)

N_HOUR, N_WD, N_HOL = 24, 7, 2
CAT_W = N_HOUR + N_WD + N_HOL
EMB = (8, 4, 2)

HIST_NAMES = [
    "load",
    *[f"obs_{v}" for v in WEATHER_VARS],
    "holiday_share",
    "doy_sin",
    "doy_cos",
]
FUT_NAMES = [
    *[f"fc_{v}" for v in WEATHER_VARS],
    "hdd",
    "cdd",
    "holiday_share",
    "bridge_day",
    "christmas_period",
    "doy_sin",
    "doy_cos",
    "lag168",
    "latest_same_hour",
    "lag48",
    "last_obs_load",
    "load_mean24",
    "temp_obs_mean24",
]
LATEST_IDX = FUT_NAMES.index("latest_same_hour")
Q_DEFAULT = (0.1, 0.5, 0.9)


@dataclass
class Scaler:
    load_mean: float
    load_std: float
    w_mean: list[float]
    w_std: list[float]
    hdd_mean: float
    hdd_std: float
    cdd_mean: float
    cdd_std: float

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Scaler:
        return cls(**d)


def fit_scaler(zf: ZoneFrame, train_end_pos: int) -> Scaler:
    """Statistics from the training period only (positions < train_end_pos)."""
    load = zf.load[:train_end_pos]
    obs = zf.obs[:train_end_pos]
    temp = obs[:, 0]
    hdd = np.maximum(zf.cfg.features.hdd_base_c - temp, 0)
    cdd = np.maximum(temp - zf.cfg.features.cdd_base_c, 0)
    sd = lambda a: float(np.nanstd(a)) or 1.0  # noqa: E731
    return Scaler(
        float(np.nanmean(load)),
        sd(load),
        [float(np.nanmean(obs[:, j])) for j in range(obs.shape[1])],
        [sd(obs[:, j]) for j in range(obs.shape[1])],
        float(np.nanmean(hdd)),
        sd(hdd),
        float(np.nanmean(cdd)),
        sd(cdd),
    )


def train_end_position(zf: ZoneFrame, train_end: str) -> int:
    import pandas as pd

    t = pd.Timestamp(train_end, tz="UTC") + pd.Timedelta(days=1)
    return int(zf.index.searchsorted(t))


class HourlyBase:
    """Per-hour standardised history features for a zone (T, F) plus categorical indices."""

    def __init__(self, zf: ZoneFrame, sc: Scaler, use_weather: bool = True):
        T = zf.T
        h = np.zeros((T, len(HIST_NAMES)), dtype=np.float32)
        h[:, 0] = (zf.load - sc.load_mean) / sc.load_std
        w = (zf.obs - np.array(sc.w_mean)) / np.array(sc.w_std)
        h[:, 1 : 1 + len(WEATHER_VARS)] = w if use_weather else 0.0
        h[:, 1 + len(WEATHER_VARS)] = zf.hol
        ang = 2 * np.pi * zf.doy / 365.25
        h[:, 2 + len(WEATHER_VARS)] = np.sin(ang)
        h[:, 3 + len(WEATHER_VARS)] = np.cos(ang)
        self.cont = torch.from_numpy(np.nan_to_num(h, nan=0.0))
        cat = np.stack([zf.hour, zf.weekday, (zf.hol >= 0.5).astype(int)], axis=1)
        self.cat = torch.from_numpy(cat.astype(np.int64))


def fut_features(
    zf: ZoneFrame, o: Origins, sc: Scaler, use_weather: bool = True
) -> tuple[torch.Tensor, torch.Tensor]:
    n = len(o)
    cols = []
    for v in WEATHER_VARS:
        j = WEATHER_VARS.index(v)
        a = o.fut["w_" + v]
        cols.append((a - sc.w_mean[j]) / sc.w_std[j] if use_weather else np.zeros_like(a))
    cols.append((o.fut["hdd"] - sc.hdd_mean) / sc.hdd_std if use_weather else np.zeros((n, 24)))
    cols.append((o.fut["cdd"] - sc.cdd_mean) / sc.cdd_std if use_weather else np.zeros((n, 24)))
    cols += [o.fut["holiday_share"], o.fut["bridge_day"], o.fut["christmas_period"]]
    ang = 2 * np.pi * o.fut["doy"] / 365.25
    cols += [np.sin(ang), np.cos(ang)]
    ls = lambda a: (a - sc.load_mean) / sc.load_std  # noqa: E731
    cols += [ls(o.fut["lag168"]), ls(o.fut["latest_same_hour"]), ls(o.fut["lag48"])]
    for k in ("last_obs_load", "load_mean24"):
        cols.append(np.repeat(ls(o.ctx[k])[:, None], 24, axis=1))
    j0 = 0
    cols.append(
        np.repeat(((o.ctx["temp_obs_mean24"] - sc.w_mean[j0]) / sc.w_std[j0])[:, None], 24, axis=1)
    )
    x = np.stack(cols, axis=-1).astype(np.float32)
    x = np.nan_to_num(x, nan=0.0)
    cat = np.stack(
        [
            zf.hour[o.tpos],
            zf.weekday[o.tpos],
            (zf.hol[o.tpos] >= 0.5).astype(int),
        ],
        axis=-1,
    ).astype(np.int64)
    return torch.from_numpy(x), torch.from_numpy(cat)


def onehot(
    cat: torch.Tensor, zone_idx: torch.Tensor | None, n_zones: int, use_cal: bool
) -> torch.Tensor:
    """cat (...,3) long -> (..., CAT_W [+ n_zones]) float one-hot (zeros if calendar disabled)."""
    if use_cal:
        parts = [
            F.one_hot(cat[..., 0], N_HOUR),
            F.one_hot(cat[..., 1], N_WD),
            F.one_hot(cat[..., 2], N_HOL),
        ]
    else:
        parts = [torch.zeros(*cat.shape[:-1], CAT_W, dtype=torch.long)]
    oh = torch.cat(parts, dim=-1).float()
    if n_zones > 1:
        assert zone_idx is not None
        z = F.one_hot(zone_idx, n_zones).float()
        while z.dim() < oh.dim():
            z = z.unsqueeze(1)
        z = z.expand(*oh.shape[:-1], n_zones)
        oh = torch.cat([oh, z], dim=-1)
    return oh


class QuantileLSTM(nn.Module):
    def __init__(
        self,
        n_hist: int,
        n_fut: int,
        hidden: int = 64,
        layers: int = 1,
        dropout: float = 0.1,
        n_zones: int = 1,
        n_q: int = 3,
    ):
        super().__init__()
        self.n_zones = n_zones
        self.emb_h = nn.Linear(N_HOUR, EMB[0], bias=False)
        self.emb_w = nn.Linear(N_WD, EMB[1], bias=False)
        self.emb_o = nn.Linear(N_HOL, EMB[2], bias=False)
        self.emb_z = nn.Linear(n_zones, 4, bias=False) if n_zones > 1 else None
        e = sum(EMB) + (4 if n_zones > 1 else 0)
        self.lstm = nn.LSTM(
            n_hist + e, hidden, layers, batch_first=True, dropout=dropout if layers > 1 else 0.0
        )
        self.dec = nn.Sequential(
            nn.Linear(hidden + n_fut + e, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_q),
        )

    def embed(self, oh: torch.Tensor) -> torch.Tensor:
        a, b = N_HOUR, N_HOUR + N_WD
        parts = [
            self.emb_h(oh[..., :a]),
            self.emb_w(oh[..., a:b]),
            self.emb_o(oh[..., b:CAT_W]),
        ]
        if self.emb_z is not None:
            parts.append(self.emb_z(oh[..., CAT_W:]))
        return torch.cat(parts, dim=-1)

    def forward(
        self, hist_c: torch.Tensor, hist_oh: torch.Tensor, fut_c: torch.Tensor, fut_oh: torch.Tensor
    ) -> torch.Tensor:
        x = torch.cat([hist_c, self.embed(hist_oh)], dim=-1)
        out, _ = self.lstm(x)
        h = out[:, -1]
        hf = h.unsqueeze(1).expand(-1, fut_c.shape[1], -1)
        d = self.dec(torch.cat([hf, fut_c, self.embed(fut_oh)], dim=-1))  # (B,24,3)
        base = fut_c[..., LATEST_IDX]
        med = base + d[..., 1]
        lo = med - F.softplus(d[..., 0])
        hi = med + F.softplus(d[..., 2])
        return torch.stack([lo, med, hi], dim=-1)


def pinball_loss(pred: torch.Tensor, y: torch.Tensor, qs: tuple[float, ...]) -> torch.Tensor:
    t = torch.tensor(qs, dtype=pred.dtype).view(1, 1, -1)
    d = y.unsqueeze(-1) - pred
    return torch.maximum(t * d, (t - 1.0) * d).mean()


class ZoneData:
    """Everything needed to batch one zone: base history, per-origin future features, targets."""

    def __init__(
        self,
        zf: ZoneFrame,
        o: Origins,
        sc: Scaler,
        zone_idx: int = 0,
        use_weather: bool = True,
        base: HourlyBase | None = None,
    ):
        self.base = base or HourlyBase(zf, sc, use_weather)
        m = o.valid
        self.mask = m
        self.last = torch.from_numpy(o.sched["last_obs_pos"].to_numpy()[m].astype(np.int64))
        fc, fk = fut_features(zf, o, sc, use_weather)
        self.fut_c, self.fut_k = fc[m], fk[m]
        y = (o.y[m] - sc.load_mean) / sc.load_std
        self.y = torch.from_numpy(np.nan_to_num(y, nan=0.0).astype(np.float32))
        self.zone_idx = zone_idx
        self.n = int(m.sum())
        self.sc = sc


def make_batch(
    data: ZoneData, idx: torch.Tensor, lookback: int, n_zones: int, use_cal: bool
) -> tuple[torch.Tensor, ...]:
    last = data.last[idx]
    pos = last.unsqueeze(1) - torch.arange(lookback - 1, -1, -1).unsqueeze(0)
    pos = pos.clamp(min=0)
    hc = data.base.cont[pos]
    hk = data.base.cat[pos]
    z = torch.full((len(idx),), data.zone_idx, dtype=torch.long)
    hoh = onehot(hk, z, n_zones, use_cal)
    foh = onehot(data.fut_k[idx], z, n_zones, use_cal)
    return hc, hoh, data.fut_c[idx], foh, data.y[idx]


class LSTMForecaster(Forecaster):
    name = "lstm"

    def __init__(
        self,
        cfg: Config,
        seed: int = 0,
        hp: dict[str, Any] | None = None,
        use_weather: bool = True,
        use_cal_emb: bool = True,
        lookback: int | None = None,
        phase_offsets: tuple[int, ...] = (0, 6, 12, 18),
        train_exclude: list[tuple[str, str]] | None = None,
        label: str | None = None,
    ):
        self.cfg = cfg
        self.seed = seed
        self.hp = {**cfg.lstm.model_dump(), **(hp or {})}
        self.use_weather = use_weather
        self.use_cal_emb = use_cal_emb
        self.lookback = lookback or int(self.hp["lookback"])
        self.phase_offsets = phase_offsets
        self.train_exclude = train_exclude or []
        self.qs = tuple(self.hp["quantiles"])
        self.model: QuantileLSTM | None = None
        self.scaler: Scaler | None = None
        self.history: list[dict[str, float]] = []
        self.best_epoch = -1
        self.train_seconds = 0.0
        self.name = label or "lstm"

    # -- construction ------------------------------------------------------------------
    def _new_model(self) -> QuantileLSTM:
        return QuantileLSTM(
            len(HIST_NAMES),
            len(FUT_NAMES),
            int(self.hp["hidden"]),
            int(self.hp["layers"]),
            float(self.hp["dropout"]),
            n_zones=1,
        )

    def _batches(self, d: ZoneData, bs: int, shuffle: bool, gen: torch.Generator | None):
        order = torch.randperm(d.n, generator=gen) if shuffle else torch.arange(d.n)
        for i in range(0, d.n, bs):
            yield order[i : i + bs]

    def _filter_excluded(self, o: Origins) -> Origins:
        if not self.train_exclude:
            return o
        dates = o.dates
        keep = np.ones(len(o), dtype=bool)
        for s, e in self.train_exclude:
            keep &= ~((dates >= s) & (dates <= e))
        o2 = copy.copy(o)
        o2.valid = o.valid & keep
        return o2

    def fit(self, zf: ZoneFrame, train: Origins, val: Origins | None = None) -> LSTMForecaster:
        assert val is not None, "early stopping needs validation origins"
        set_seed(self.seed)
        torch.set_num_threads(max(1, torch.get_num_threads()))
        te = train_end_position(zf, self.cfg.splits.train_end)
        self.scaler = fit_scaler(zf, te)
        base = HourlyBase(zf, self.scaler, self.use_weather)
        dtr = ZoneData(
            zf, self._filter_excluded(train), self.scaler, use_weather=self.use_weather, base=base
        )
        dva = ZoneData(zf, val, self.scaler, use_weather=self.use_weather, base=base)
        model = self._new_model()
        opt = torch.optim.AdamW(
            model.parameters(), lr=float(self.hp["lr"]), weight_decay=float(self.hp["weight_decay"])
        )
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=2)
        gen = torch.Generator().manual_seed(self.seed)
        best = float("inf")
        best_state = copy.deepcopy(model.state_dict())
        bad = 0
        t0 = time.time()
        bs = int(self.hp["batch_size"])
        for epoch in range(int(self.hp["max_epochs"])):
            model.train()
            tot, cnt = 0.0, 0
            for idx in self._batches(dtr, bs, True, gen):
                hc, hoh, fc, foh, y = make_batch(dtr, idx, self.lookback, 1, self.use_cal_emb)
                pred = model(hc, hoh, fc, foh)
                loss = pinball_loss(pred, y, self.qs)
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), float(self.hp["clip"]))
                opt.step()
                tot += loss.item() * len(idx)
                cnt += len(idx)
            vloss = self._eval_loss(model, dva)
            sched.step(vloss)
            self.history.append({"epoch": epoch, "train_loss": tot / cnt, "val_loss": vloss})
            log.info("%s ep %d train %.4f val %.4f", zf.zone, epoch, tot / cnt, vloss)
            if vloss < best - 1e-5:
                best, bad, self.best_epoch = vloss, 0, epoch
                best_state = copy.deepcopy(model.state_dict())
            else:
                bad += 1
                if bad >= int(self.hp["patience"]):
                    break
        model.load_state_dict(best_state)
        model.eval()
        self.model = model
        self.train_seconds = time.time() - t0
        self.best_val_loss = best
        return self

    @torch.no_grad()
    def _eval_loss(self, model: QuantileLSTM, d: ZoneData) -> float:
        model.eval()
        tot, cnt = 0.0, 0
        for idx in self._batches(d, 512, False, None):
            hc, hoh, fc, foh, y = make_batch(d, idx, self.lookback, 1, self.use_cal_emb)
            tot += float(pinball_loss(model(hc, hoh, fc, foh), y, self.qs)) * len(idx)
            cnt += len(idx)
        return tot / max(cnt, 1)

    # -- inference ---------------------------------------------------------------------
    @torch.no_grad()
    def _raw(self, zf: ZoneFrame, o: Origins) -> np.ndarray:
        assert self.model is not None and self.scaler is not None
        full = copy.copy(o)
        full.valid = np.ones(len(o), dtype=bool)  # predict for every row; callers mask
        d = ZoneData(zf, full, self.scaler, use_weather=self.use_weather)
        outs = []
        self.model.eval()
        for idx in self._batches(d, 512, False, None):
            hc, hoh, fc, foh, _ = make_batch(d, idx, self.lookback, 1, self.use_cal_emb)
            outs.append(self.model(hc, hoh, fc, foh).numpy())
        arr = np.concatenate(outs) if outs else np.zeros((0, 24, 3), dtype=np.float32)
        return arr.astype(np.float64) * self.scaler.load_std + self.scaler.load_mean

    def predict(self, zf: ZoneFrame, o: Origins) -> np.ndarray:
        return self._raw(zf, o)[..., 1]

    def predict_quantiles(
        self, zf: ZoneFrame, o: Origins, quantiles: tuple[float, ...] = Q_DEFAULT
    ) -> dict[float, np.ndarray]:
        raw = self._raw(zf, o)
        return {q: raw[..., self.qs.index(q)] for q in quantiles}

    # -- persistence -------------------------------------------------------------------
    def save(self, folder: Path) -> None:
        assert self.model is not None and self.scaler is not None
        folder.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), folder / "model.pt")
        (folder / "meta.json").write_text(
            json.dumps(
                {
                    "scaler": self.scaler.to_dict(),
                    "hp": self.hp,
                    "lookback": self.lookback,
                    "use_weather": self.use_weather,
                    "use_cal_emb": self.use_cal_emb,
                    "best_epoch": self.best_epoch,
                    "history": self.history,
                    "seed": self.seed,
                    "train_seconds": self.train_seconds,
                    "hist_names": HIST_NAMES,
                    "fut_names": FUT_NAMES,
                }
            )
        )

    @classmethod
    def load(cls, folder: Path, cfg: Config) -> LSTMForecaster:
        meta = json.loads((folder / "meta.json").read_text())
        f = cls(
            cfg,
            seed=meta["seed"],
            hp=meta["hp"],
            use_weather=meta["use_weather"],
            use_cal_emb=meta["use_cal_emb"],
            lookback=meta["lookback"],
        )
        f.scaler = Scaler.from_dict(meta["scaler"])
        f.model = f._new_model()
        f.model.load_state_dict(torch.load(folder / "model.pt", map_location="cpu"))
        f.model.eval()
        f.history = meta["history"]
        f.best_epoch = meta["best_epoch"]
        f.train_seconds = meta.get("train_seconds", 0.0)
        return f

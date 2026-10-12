"""One LSTM trained across the four control areas with a zone embedding."""

from __future__ import annotations

import copy
import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from energy.config import Config, set_seed
from energy.evaluation.backtest import Context
from energy.features import Origins, ZoneFrame
from energy.models.base import Forecaster
from energy.models.lstm import (
    FUT_NAMES,
    HIST_NAMES,
    HourlyBase,
    QuantileLSTM,
    Scaler,
    ZoneData,
    fit_scaler,
    make_batch,
    pinball_loss,
    train_end_position,
)

log = logging.getLogger(__name__)


class PooledLSTM:
    """Trains on all zones jointly. Each zone keeps its own standardisation (training period
    statistics of that zone); the zone enters through a learned embedding."""

    name = "pooled_lstm"

    def __init__(
        self,
        cfg: Config,
        zones: list[str],
        seed: int = 0,
        hp: dict[str, Any] | None = None,
        phase_offsets: tuple[int, ...] = (0, 6, 12, 18),
    ):
        self.cfg = cfg
        self.zones = zones
        self.seed = seed
        self.hp = {**cfg.lstm.model_dump(), **(hp or {})}
        self.lookback = int(self.hp["lookback"])
        self.qs = tuple(self.hp["quantiles"])
        self.phase_offsets = phase_offsets
        self.model: QuantileLSTM | None = None
        self.scalers: dict[str, Scaler] = {}
        self.history: list[dict[str, float]] = []
        self.train_seconds = 0.0

    def _new_model(self) -> QuantileLSTM:
        return QuantileLSTM(
            len(HIST_NAMES),
            len(FUT_NAMES),
            int(self.hp["hidden"]),
            int(self.hp["layers"]),
            float(self.hp["dropout"]),
            n_zones=len(self.zones),
        )

    def _data(self, ctx: Context, o: Origins, zone: str) -> ZoneData:
        return ZoneData(ctx.zf, o, self.scalers[zone], zone_idx=self.zones.index(zone))

    def fit(self, ctxs: dict[str, Context]) -> PooledLSTM:
        set_seed(self.seed)
        train: dict[str, ZoneData] = {}
        val: dict[str, ZoneData] = {}
        for z in self.zones:
            ctx = ctxs[z]
            self.scalers[z] = fit_scaler(
                ctx.zf, train_end_position(ctx.zf, self.cfg.splits.train_end)
            )
            base = HourlyBase(ctx.zf, self.scalers[z])
            tr = ZoneData(
                ctx.zf,
                ctx.origins("train", offsets=self.phase_offsets),
                self.scalers[z],
                self.zones.index(z),
                base=base,
            )
            va = ZoneData(
                ctx.zf, ctx.origins("val"), self.scalers[z], self.zones.index(z), base=base
            )
            train[z], val[z] = tr, va
        model = self._new_model()
        opt = torch.optim.AdamW(
            model.parameters(), lr=float(self.hp["lr"]), weight_decay=float(self.hp["weight_decay"])
        )
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=2)
        gen = torch.Generator().manual_seed(self.seed)
        nz = len(self.zones)
        bs = int(self.hp["batch_size"])
        best, bad, best_state = float("inf"), 0, copy.deepcopy(model.state_dict())
        t0 = time.time()
        for epoch in range(int(self.hp["max_epochs"])):
            model.train()
            batches = []
            for z in self.zones:
                order = torch.randperm(train[z].n, generator=gen)
                batches += [(z, order[i : i + bs]) for i in range(0, train[z].n, bs)]
            perm = torch.randperm(len(batches), generator=gen).tolist()
            tot, cnt = 0.0, 0
            for k in perm:
                z, idx = batches[k]
                hc, hoh, fc, foh, y = make_batch(train[z], idx, self.lookback, nz, True)
                loss = pinball_loss(model(hc, hoh, fc, foh), y, self.qs)
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), float(self.hp["clip"]))
                opt.step()
                tot += loss.item() * len(idx)
                cnt += len(idx)
            vloss = self._val_loss(model, val)
            sched.step(vloss)
            self.history.append({"epoch": epoch, "train_loss": tot / cnt, "val_loss": vloss})
            log.info("pooled ep %d train %.4f val %.4f", epoch, tot / cnt, vloss)
            if vloss < best - 1e-5:
                best, bad = vloss, 0
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
    def _val_loss(self, model: QuantileLSTM, val: dict[str, ZoneData]) -> float:
        model.eval()
        tot, cnt = 0.0, 0
        for d in val.values():
            for i in range(0, d.n, 512):
                idx = torch.arange(i, min(i + 512, d.n))
                hc, hoh, fc, foh, y = make_batch(d, idx, self.lookback, len(self.zones), True)
                tot += float(pinball_loss(model(hc, hoh, fc, foh), y, self.qs)) * len(idx)
                cnt += len(idx)
        return tot / max(cnt, 1)

    @torch.no_grad()
    def raw(self, zone: str, zf: ZoneFrame, o: Origins) -> np.ndarray:
        assert self.model is not None
        sc = self.scalers[zone]
        full = copy.copy(o)
        full.valid = np.ones(len(o), dtype=bool)
        d = ZoneData(zf, full, sc, zone_idx=self.zones.index(zone))
        outs = []
        for i in range(0, d.n, 512):
            idx = torch.arange(i, min(i + 512, d.n))
            hc, hoh, fc, foh, _ = make_batch(d, idx, self.lookback, len(self.zones), True)
            outs.append(self.model(hc, hoh, fc, foh).numpy())
        arr = np.concatenate(outs) if outs else np.zeros((0, 24, 3), dtype=np.float32)
        return arr.astype(np.float64) * sc.load_std + sc.load_mean

    def view(self, zone: str) -> PooledView:
        return PooledView(self, zone)

    def save(self, folder: Path) -> None:
        assert self.model is not None
        folder.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), folder / "model.pt")
        (folder / "meta.json").write_text(
            json.dumps(
                {
                    "zones": self.zones,
                    "hp": self.hp,
                    "seed": self.seed,
                    "scalers": {z: s.to_dict() for z, s in self.scalers.items()},
                    "history": self.history,
                    "train_seconds": self.train_seconds,
                }
            )
        )

    @classmethod
    def load(cls, folder: Path, cfg: Config) -> PooledLSTM:
        meta = json.loads((folder / "meta.json").read_text())
        p = cls(cfg, meta["zones"], meta["seed"], meta["hp"])
        p.scalers = {z: Scaler.from_dict(s) for z, s in meta["scalers"].items()}
        p.model = p._new_model()
        p.model.load_state_dict(torch.load(folder / "model.pt", map_location="cpu"))
        p.model.eval()
        p.history = meta["history"]
        return p


class PooledView(Forecaster):
    """Single-zone view of a pooled model, so it plugs into the common backtest runner."""

    def __init__(self, pooled: PooledLSTM, zone: str):
        self.pooled, self.zone = pooled, zone
        self.name = "pooled_lstm"

    def fit(self, zf: ZoneFrame, train: Origins, val: Origins | None = None) -> PooledView:
        return self

    def predict(self, zf: ZoneFrame, o: Origins) -> np.ndarray:
        return self.pooled.raw(self.zone, zf, o)[..., 1]

    def predict_quantiles(
        self, zf: ZoneFrame, o: Origins, quantiles: tuple[float, ...] = (0.1, 0.5, 0.9)
    ) -> dict[float, np.ndarray]:
        raw = self.pooled.raw(self.zone, zf, o)
        return {q: raw[..., self.pooled.qs.index(q)] for q in quantiles}

"""The refresh job: fetch fresh inputs, check them, forecast, explain and publish to the store.

Forecasts are never published from stale or failing inputs. In that case only the refresh
status is updated, and the API keeps serving the last good forecast with a staleness flag.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from energy.config import ALL_REGIONS, Config, load_config
from energy.evaluation.runner import read_calibration
from energy.explain.shap_lstm import explain_inputs
from energy.features import build_origins
from energy.models.lstm import LSTMForecaster, ZoneData, make_batch
from energy.serving.live import (
    build_live_frame,
    fetch_live_load,
    fetch_live_weather,
    pick_issue,
)
from energy.store.base import ArtifactStore
from energy.training.manifest import validate_manifest

log = logging.getLogger(__name__)

MODEL_NAME = "lstm"
TZ = "Europe/Berlin"


def publish_models(store: ArtifactStore, cfg: Config, zones: list[str] | None = None) -> list[str]:
    """Upload validated LSTM artifacts and serving configuration to the store."""
    calib = read_calibration(cfg)
    done = []
    for z in zones or ALL_REGIONS:
        folder = cfg.path("artifacts") / "models" / z / MODEL_NAME
        manifest = validate_manifest(folder)
        store.upload_dir(folder, f"models/{z}/{MODEL_NAME}")
        offset = calib.get(z, {}).get(MODEL_NAME, {}).get("archived")
        if offset is None:
            offset = calib.get(z, {}).get(MODEL_NAME, {}).get("reanalysis", 0.0)
        store.put_json(
            f"models/{z}/serving.json",
            {
                "model": MODEL_NAME,
                "cqr_offset_mw": float(offset),
                "calibrated_on": "validation 2023 (archived-forecast weather where available)",
                "model_created_utc": manifest["created_utc"],
                "config_hash": manifest["config_hash"],
            },
        )
        done.append(z)
    return done


def load_serving_model(
    store: ArtifactStore, zone: str, cache: Path
) -> tuple[LSTMForecaster, dict[str, Any], dict[str, Any], list[torch.Tensor]]:
    folder = cache / zone / MODEL_NAME
    if folder.exists():
        shutil.rmtree(folder)
    store.download_dir(f"models/{zone}/{MODEL_NAME}", folder)
    manifest = validate_manifest(folder)  # raises when missing or inconsistent
    serving = store.get_json(f"models/{zone}/serving.json")
    if serving["config_hash"] != manifest["config_hash"]:
        raise ValueError(f"serving.json does not match manifest for {zone}")
    cfg = load_config()
    f = LSTMForecaster.load(folder, cfg)
    bg = torch.load(folder / "background.pt", map_location="cpu")
    return f, manifest, serving, bg


def check_inputs(
    zf: Any, o: Any, f: LSTMForecaster, issue: pd.Timestamp, now: pd.Timestamp
) -> list[str]:
    """Return a list of reasons to refuse publishing (empty when the inputs are fine)."""
    problems = []
    assert f.scaler is not None
    last = int(o.sched["last_obs_pos"].iloc[0])
    lo = f.scaler.load_mean - 6 * f.scaler.load_std
    hi = f.scaler.load_mean + 6 * f.scaler.load_std
    window = zf.load[last - 167 : last + 1]
    if np.isnan(window).any():
        problems.append("load history window has missing values")
    elif window.min() < lo or window.max() > hi:
        problems.append("load history outside plausible range")
    if (now - issue) > pd.Timedelta(hours=36):
        problems.append("latest usable issue time is more than 36 hours old")
    t0 = int(o.sched["tstart_pos"].iloc[0])
    if np.isnan(zf.fc[t0 : t0 + 24, 0]).any():
        problems.append("weather forecast missing for target day")
    return problems


def forecast_one(
    zone: str,
    f: LSTMForecaster,
    manifest: dict[str, Any],
    serving: dict[str, Any],
    bg: list[torch.Tensor],
    zf: Any,
    issue: pd.Timestamp,
    target: pd.Timestamp,
    cfg: Config,
    explain: bool = True,
    nsamples: int = 100,
) -> dict[str, Any]:
    d = target.strftime("%Y-%m-%d")
    o = build_origins(zf, cfg, d, d, "archived", require_target=False)
    raw = f._raw(zf, o)[0]  # (24, 3) MW
    off = float(serving["cqr_offset_mw"])
    lo, med, hi = raw[:, 0] - off, raw[:, 1], raw[:, 2] + off
    ts = zf.index[o.tpos[0]]
    points = [
        {
            "ts_utc": t.isoformat(),
            "hour_local": int(t.tz_convert(TZ).hour),
            "p10": float(lo[i]),
            "p50": float(med[i]),
            "p90": float(hi[i]),
        }
        for i, t in enumerate(ts)
    ]
    out: dict[str, Any] = {
        "zone": zone,
        "issue_time": issue.isoformat(),
        "issue_time_local": issue.tz_convert(TZ).isoformat(),
        "target_date": d,
        "model": MODEL_NAME,
        "model_version": f"{manifest['created_utc']}|{manifest['git_commit']}|{manifest['config_hash']}",
        "input_data_timestamp": zf.index[int(o.sched["last_obs_pos"].iloc[0])].isoformat(),
        "weather_source": "open-meteo live forecast API (population-weighted, approximate)",
        "calibration_offset_mw": off,
        "points": points,
        "explanation_available": explain,
        "created_utc": pd.Timestamp.now(tz="UTC").isoformat(),
    }
    if explain:
        d_zone = ZoneData(zf, o, f.scaler, use_weather=f.use_weather)  # type: ignore[arg-type]
        idx = torch.arange(d_zone.n)
        hc, hoh, fc, foh, _ = make_batch(d_zone, idx, f.lookback, 1, f.use_cal_emb)
        expl: dict[str, list[dict[str, float | str]]] = {}
        for h in range(24):
            e = explain_inputs(f, [hc, hoh, fc, foh], bg, [h], nsamples)
            a = e.attr[0]
            order = np.argsort(-np.abs(a))[:8]
            expl[str(h)] = [{"feature": e.names[i], "attribution_mw": float(a[i])} for i in order]
            expl[str(h)].append({"feature": "_base_value", "attribution_mw": float(e.base_value)})
        out["explanations"] = expl
    return out


def refresh(
    store: ArtifactStore,
    cfg: Config | None = None,
    now: pd.Timestamp | None = None,
    zones: list[str] | None = None,
    fetch: bool = True,  # refetch the latest SMARD load chunks
    explain: bool = True,
    weather: dict[str, pd.DataFrame] | None = None,
    shap_samples: int = 100,
) -> dict[str, Any]:
    cfg = cfg or load_config()
    now = now or pd.Timestamp.now(tz="UTC")
    zones = zones or ALL_REGIONS
    cache = cfg.path("artifacts") / "cache" / "models"
    status: dict[str, Any] = {"run_utc": now.isoformat(), "zones": {}, "published": False}
    if fetch:
        fetch_live_load(cfg)
    if weather is None:
        # Weather has no useful cache for the target day, so it is always fetched live
        # unless the caller supplies it.
        weather = fetch_live_weather(cfg)
    for z in zones:
        zs: dict[str, Any] = {"ok": False, "reasons": []}
        status["zones"][z] = zs
        try:
            f, manifest, serving, bg = load_serving_model(store, z, cache)
        except Exception as e:  # missing or inconsistent artifacts: refuse
            zs["reasons"].append(f"artifact problem: {e}")
            continue
        zf = build_live_frame(z, cfg, weather, now)
        issue, target = pick_issue(zf, now, cfg)
        if issue is None or target is None:
            zs["reasons"].append("no issue slot with complete inputs in the last 4 days")
            continue
        o = build_origins(
            zf,
            cfg,
            target.strftime("%Y-%m-%d"),
            target.strftime("%Y-%m-%d"),
            "archived",
            require_target=False,
        )
        problems = check_inputs(zf, o, f, issue, now)
        if problems:
            zs["reasons"] += problems
            continue
        fc = forecast_one(
            z, f, manifest, serving, bg, zf, issue, target, cfg, explain, shap_samples
        )
        key_date = target.strftime("%Y-%m-%d")
        store.put_json(f"forecasts/{z}/{key_date}.json", fc)
        store.put_json(f"forecasts/{z}/latest.json", fc)
        hist = zf.load[: int(o.sched["last_obs_pos"].iloc[0]) + 1][-24 * 14 :]
        hidx = zf.index[: len(zf.load)][: int(o.sched["last_obs_pos"].iloc[0]) + 1][-24 * 14 :]
        store.put_json(
            f"actuals/{z}.json",
            {
                "zone": z,
                "ts_utc": [t.isoformat() for t in hidx],
                "load_mw": [None if np.isnan(v) else float(v) for v in hist],
            },
        )
        zs.update(ok=True, target_date=key_date, issue_time=issue.isoformat())
    status["published"] = any(v["ok"] for v in status["zones"].values())
    store.put_json("status/refresh.json", status)
    if status["published"]:
        store.put_json("status/last_good.json", status)
    return status

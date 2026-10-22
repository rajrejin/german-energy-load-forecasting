"""Hardware and runtime records: results/environment.json and results/runtime.json."""

from __future__ import annotations

import json
import os
import platform
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd
import psutil

from energy.config import Config, load_config
from energy.training.manifest import library_versions


def _cpu_name() -> str:
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"],
            capture_output=True,
            text=True,
            timeout=20,
        ).stdout.strip()
        if out:
            return out
    except Exception:
        pass
    return platform.processor() or "unknown"


def write_environment(cfg: Config | None = None) -> dict[str, Any]:
    cfg = cfg or load_config()
    try:
        import torch

        torch_threads = torch.get_num_threads()
        cuda = torch.cuda.is_available()
    except ImportError:  # pragma: no cover
        torch_threads, cuda = None, False
    env = {
        "platform": platform.platform(),
        "cpu": _cpu_name(),
        "logical_cpus": os.cpu_count(),
        "physical_cpus": psutil.cpu_count(logical=False),
        "ram_gb": round(psutil.virtual_memory().total / 2**30, 2),
        "gpu_available": cuda,
        "torch_threads_default": torch_threads,
        "libraries": library_versions(),
        "seed": cfg.seed,
        "config_hash": cfg.hash(),
    }
    (cfg.path("results") / "environment.json").write_text(json.dumps(env, indent=2))
    return env


def write_runtime(cfg: Config | None = None) -> dict[str, Any]:
    """Fit times from artifact manifests, search times from the HPO logs, Dask timing."""
    cfg = cfg or load_config()
    models = cfg.path("artifacts") / "models"
    fit: dict[str, dict[str, float]] = {}
    for m in sorted(models.glob("*/*/manifest.json")):
        d = json.loads(m.read_text())
        sec = d.get("val_metrics", {}).get("fit_seconds")
        if sec is not None:
            fit.setdefault(m.parent.parent.name, {})[m.parent.name] = round(float(sec), 1)
    hpo = {}
    for p in sorted((cfg.path("results") / "hpo").glob("*_DE.csv")):
        df = pd.read_csv(p)
        hpo[p.stem] = {
            "trials": len(df),
            "total_seconds": round(float(df["seconds"].sum()), 1) if "seconds" in df else None,
        }
    pooled = models / "pooled" / "pooled_lstm" / "meta.json"
    if pooled.exists():
        pm = json.loads(pooled.read_text())
        fit.setdefault("pooled", {})["pooled_lstm"] = round(float(pm.get("train_seconds", 0.0)), 1)
    out: dict[str, Any] = {"fit_seconds_by_zone_and_model": fit, "hpo": hpo}
    pt = cfg.path("results") / "parallel_timing.json"
    if pt.exists():
        d = json.loads(pt.read_text())
        out["dask_benchmark"] = {
            k: d[k]
            for k in (
                "sequential_1_thread",
                "sequential_all_threads",
                "parallel",
                "speedup_vs_seq_1_thread",
                "speedup_vs_seq_all_threads",
            )
            if k in d
        }
        for k in ("sequential_1_thread", "sequential_all_threads", "parallel"):
            out["dask_benchmark"][k] = {
                kk: vv for kk, vv in out["dask_benchmark"][k].items() if kk != "tasks"
            }
    be = cfg.path("results") / "briefing_eval.json"
    if be.exists():
        out["briefing_seconds_per_briefing"] = json.loads(be.read_text()).get(
            "mean_seconds_per_briefing"
        )
    (cfg.path("results") / "runtime.json").write_text(json.dumps(out, indent=2))
    return out


__all__ = ["write_environment", "write_runtime", "Path"]

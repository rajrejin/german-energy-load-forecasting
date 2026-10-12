"""Per-zone parallel training on a local Dask cluster, with timing and memory measurement."""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import psutil

from energy.config import ZONES, Config, load_config

log = logging.getLogger(__name__)


def _train_task(
    zone: str, kind: str, seed: int, max_epochs: int, torch_threads: int, out_dir: str
) -> dict[str, Any]:
    """Runs inside a worker process (or inline for the sequential reference)."""
    import torch

    from energy.evaluation.backtest import collect, make_context
    from energy.training.train import fit_model

    torch.set_num_threads(torch_threads)
    ctx = make_context(zone)
    t0 = time.time()
    hp = {"max_epochs": max_epochs} if kind == "lstm" else None
    model, met = fit_model(kind, ctx, seed=seed, hp=hp)
    val = collect(model, ctx, ctx.origins("val"))
    path = Path(out_dir) / f"{zone}_{kind}.npy"
    np.save(path, val.p)
    return {
        "zone": zone,
        "seconds": time.time() - t0,
        "val_mae": met["mae"],
        "worker_rss_mb": psutil.Process(os.getpid()).memory_info().rss / 2**20,
        "pred_path": str(path),
    }


class PeakMemory:
    """Samples the summed RSS of this process and all its children."""

    def __init__(self, interval: float = 0.5):
        self.interval = interval
        self.peak = 0.0
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        me = psutil.Process()
        while not self._stop.is_set():
            try:
                procs = [me, *me.children(recursive=True)]
                total = sum(p.memory_info().rss for p in procs if p.is_running())
                self.peak = max(self.peak, total / 2**20)
            except psutil.Error:
                pass
            time.sleep(self.interval)

    def __enter__(self) -> PeakMemory:
        self._t.start()
        return self

    def __exit__(self, *a: Any) -> None:
        self._stop.set()
        self._t.join()


def run_sequential(
    zones: list[str], kind: str, seed: int, max_epochs: int, torch_threads: int, out_dir: str
) -> dict[str, Any]:
    with PeakMemory() as pm:
        t0 = time.time()
        res = [_train_task(z, kind, seed, max_epochs, torch_threads, out_dir) for z in zones]
        wall = time.time() - t0
    return {"wall_seconds": wall, "peak_rss_mb": pm.peak, "tasks": res}


def run_parallel(
    zones: list[str],
    kind: str,
    seed: int,
    max_epochs: int,
    workers: int,
    threads: int,
    memory_limit: str,
    out_dir: str,
) -> dict[str, Any]:
    from dask.distributed import Client, LocalCluster

    with PeakMemory() as pm:
        t0 = time.time()
        cluster = LocalCluster(
            n_workers=workers,
            threads_per_worker=1,
            processes=True,
            memory_limit=memory_limit,
            env={"OMP_NUM_THREADS": str(threads), "MKL_NUM_THREADS": str(threads)},
            dashboard_address=None,
        )
        client = Client(cluster)
        t_up = time.time() - t0
        futs = [
            client.submit(_train_task, z, kind, seed, max_epochs, threads, out_dir, pure=False)
            for z in zones
        ]
        res = client.gather(futs)
        wall = time.time() - t0
        client.close()
        cluster.close()
    return {"wall_seconds": wall, "startup_seconds": t_up, "peak_rss_mb": pm.peak, "tasks": res}


def benchmark(
    kind: str = "lstm",
    max_epochs: int = 6,
    zones: list[str] | None = None,
    cfg: Config | None = None,
) -> dict[str, Any]:
    """Sequential (1 thread and all threads) versus parallel; writes results/parallel_timing.json."""
    cfg = cfg or load_config()
    zones = zones or ZONES
    tmp = cfg.path("processed") / "parallel_tmp"
    out: dict[str, Any] = {
        "kind": kind,
        "max_epochs": max_epochs,
        "zones": zones,
        "cpu_count": os.cpu_count(),
        "ram_gb": psutil.virtual_memory().total / 2**30,
    }
    dirs = {}
    for label in ("seq1", "seqN", "par"):
        d = tmp / label
        d.mkdir(parents=True, exist_ok=True)
        dirs[label] = str(d)
    n_threads = os.cpu_count() or 4
    out["sequential_1_thread"] = run_sequential(zones, kind, cfg.seed, max_epochs, 1, dirs["seq1"])
    out["sequential_all_threads"] = run_sequential(
        zones, kind, cfg.seed, max_epochs, n_threads, dirs["seqN"]
    )
    out["parallel"] = run_parallel(
        zones,
        kind,
        cfg.seed,
        max_epochs,
        cfg.parallel.workers,
        cfg.parallel.threads_per_worker,
        cfg.parallel.memory_limit,
        dirs["par"],
    )
    diffs = {}
    for z in zones:
        a = np.load(Path(dirs["seq1"]) / f"{z}_{kind}.npy")
        b = np.load(Path(dirs["par"]) / f"{z}_{kind}.npy")
        diffs[z] = float(np.nanmax(np.abs(a - b)))
    out["max_abs_prediction_diff_seq1_vs_parallel_mw"] = diffs
    s1 = out["sequential_1_thread"]["wall_seconds"]
    sn = out["sequential_all_threads"]["wall_seconds"]
    p = out["parallel"]["wall_seconds"]
    out["speedup_vs_seq_1_thread"] = s1 / p
    out["speedup_vs_seq_all_threads"] = sn / p
    out["parallel_workers"] = cfg.parallel.workers
    (cfg.path("results") / "parallel_timing.json").write_text(json.dumps(out, indent=2))
    return out

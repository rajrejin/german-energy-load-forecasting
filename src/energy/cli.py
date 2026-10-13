"""Command line entry point: `energy <command>`."""

from __future__ import annotations

import typer

from energy.logging_utils import setup_logging

app = typer.Typer(add_completion=False, help="German electricity demand forecaster")


@app.callback()
def _main(verbose: bool = typer.Option(False, "--verbose", "-v")) -> None:
    import logging

    setup_logging(logging.DEBUG if verbose else logging.INFO)


@app.command()
def fetch(
    what: str = typer.Option("all", help="load | weather | generation | all"),
    slice_only: bool = typer.Option(False, "--slice", help="vertical slice: 2023 onward, 1 point"),
) -> None:
    """Download raw SMARD and Open-Meteo data into the cache."""
    from energy.data.fetch import run_fetch

    run_fetch(what, slice_only)


@app.command("build-dataset")
def build_dataset() -> None:
    """Build tidy hourly tables per zone from the raw caches."""
    from energy.data.build import build_all

    build_all()


@app.command()
def quality() -> None:
    """Run data quality checks and write results/data_quality.json."""
    from energy.data.quality import run_quality

    run_quality()


@app.command()
def eda() -> None:
    """Exploratory data analysis figures and statistics."""
    from energy.data.eda import run_eda

    run_eda()


@app.command("parallel-benchmark")
def parallel_benchmark(kind: str = "lstm", max_epochs: int = 6) -> None:
    """Sequential versus Dask-parallel per-zone training (timing and memory)."""
    from energy.training.parallel import benchmark

    benchmark(kind, max_epochs)


@app.command()
def explain(zone: str = "DE", quick: bool = False) -> None:
    """SHAP explanations, plots and sanity checks for the trained LSTM."""
    from energy.explain.plots import run_explain

    run_explain(zone, quick=quick)


@app.command()
def refresh(
    store: str = typer.Option("local", help="local | azurite | azure"),
    no_fetch: bool = False,
    no_explain: bool = False,
) -> None:
    """Fetch fresh inputs, check them, forecast, explain and publish to the artifact store."""
    from energy.serving.refresh import refresh as run
    from energy.serving.settings import store_from_env

    status = run(store_from_env(store), fetch=not no_fetch, explain=not no_explain)
    typer.echo(f"published={status['published']}")
    for z, v in status["zones"].items():
        typer.echo(f"{z}: ok={v['ok']} {v.get('reasons') or ''}")


@app.command("publish-models")
def publish_models_cmd(store: str = typer.Option("local")) -> None:
    """Upload validated model artifacts and serving configuration to the store."""
    from energy.config import load_config
    from energy.serving.refresh import publish_models
    from energy.serving.settings import store_from_env

    done = publish_models(store_from_env(store), load_config())
    typer.echo(f"published models for {done}")


@app.command("publish-results")
def publish_results(store: str = typer.Option("local")) -> None:
    """Upload the result tables the dashboard shows (CSV and JSON from results/) to the store."""
    from energy.config import load_config
    from energy.serving.settings import store_from_env

    cfg = load_config()
    st = store_from_env(store)
    names = [
        "ladder.csv",
        "by_hour.csv",
        "ablations.csv",
        "significance.csv",
        "seed_spread.csv",
        "breakdowns.csv",
        "external_benchmark.csv",
        "data_quality.json",
        "explain_sanity.json",
        "parallel_timing.json",
        "briefing_eval.json",
    ]
    for n in names:
        p = cfg.path("results") / n
        if p.exists():
            st.put_bytes(f"results/{n}", p.read_bytes())
            typer.echo(f"published {n}")


@app.command()
def env() -> None:
    """Write results/environment.json and results/runtime.json."""
    from energy.environment import write_environment, write_runtime

    write_environment()
    write_runtime()


@app.command("serve-api")
def serve_api(host: str = "127.0.0.1", port: int = 8000) -> None:
    """Run the FastAPI service."""
    import uvicorn

    uvicorn.run("api.main:get_app", host=host, port=port, factory=True)


@app.command()
def ablate(
    zone: str = "DE", names: str = typer.Option("", help="comma separated, default all")
) -> None:
    """Train the ablation variants (and the pooled model with --names pooled)."""
    from energy.training.ablations import run_ablations, train_pooled

    if names == "pooled":
        train_pooled()
    else:
        run_ablations(zone, names.split(",") if names else None)


@app.command()
def backtest(
    zones: str = typer.Option("DE,50Hertz,Amprion,TenneT,TransnetBW"),
    models: str = typer.Option("", help="comma separated artifact names; default all found"),
    splits: str = typer.Option("val", help="val | val,test (test is the final, one-off run)"),
) -> None:
    """Rolling-origin backtest of trained artifacts; writes per-model predictions."""
    from energy.config import load_config
    from energy.evaluation.runner import all_artifact_names, run_backtest

    cfg = load_config()
    for z in zones.split(","):
        names = models.split(",") if models else all_artifact_names(z, cfg)
        run_backtest(z, names, tuple(splits.split(",")), cfg=cfg)


@app.command()
def report(split: str = "test") -> None:
    """Write result tables (ladder, significance, breakdowns, ablations) from stored predictions."""
    from energy.evaluation.analysis import write_all

    write_all(split=split)


@app.command()
def briefing(
    source: str = typer.Option("store", help="store (publish latest) | evaluate (test weeks)"),
    store: str = typer.Option("local"),
    weeks: int = typer.Option(0, help="number of weeks to evaluate (0 = all)"),
) -> None:
    """Weekly briefing: publish from stored forecasts, or evaluate over the test weeks."""
    from energy.briefing.evaluate import evaluate_briefings, facts_from_store, publish_briefing
    from energy.briefing.llm import get_backend
    from energy.serving.settings import store_from_env

    backend = get_backend()
    typer.echo(f"backend: {backend.name if backend else 'template (no language model available)'}")
    if source == "evaluate":
        typer.echo(str(evaluate_briefings(backend, n_weeks=weeks or None)))
        return
    st = store_from_env(store)
    facts = facts_from_store(st)
    if facts is None:
        typer.echo("fewer than 14 published daily forecasts per zone: no briefing")
        raise typer.Exit(1)
    publish_briefing(st, facts, backend)
    typer.echo("published briefing/latest.json")


@app.command()
def hpo(
    kind: str = typer.Argument(..., help="lstm | lightgbm"),
    zone: str = "DE",
    trials: int = 20,
    max_epochs: int = 10,
) -> None:
    """Random hyperparameter search on the validation split."""
    from energy.training.hpo import run_hpo

    run_hpo(kind, zone, trials, max_epochs)


@app.command()
def train(
    zones: str = typer.Option("DE,50Hertz,Amprion,TenneT,TransnetBW", help="comma separated"),
    kinds: str = typer.Option("naive_week,naive_day,lightgbm,lstm", help="comma separated"),
    seeds: str = typer.Option("0", help="comma separated seeds (LSTM only uses >1)"),
) -> None:
    """Train models per zone and save versioned artifacts."""
    from energy.training.train import train_many

    train_many(zones.split(","), kinds.split(","), [int(s) for s in seeds.split(",")])


if __name__ == "__main__":  # pragma: no cover
    app()
